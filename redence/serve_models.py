"""Serve both trained PEFT adapters through a local Ollama-compatible endpoint."""
import argparse
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic


class AdapterModel:
    def __init__(self, adapter_path, load_in_4bit=True):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        path = Path(adapter_path)
        config = PeftConfig.from_pretrained(path)
        self.metadata = json.loads((path / "redence_metadata.json").read_text())
        self.role = self.metadata["role"]
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
        kwargs = {"device_map": "auto", "torch_dtype": dtype}
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit adapter serving requires a CUDA GPU")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        base = AutoModelForImageTextToText.from_pretrained(config.base_model_name_or_path, **kwargs)
        self.model = PeftModel.from_pretrained(base, path).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(config.base_model_name_or_path) if self.role == "investigator" else None

    def chat(self, messages, max_tokens, timeout):
        import torch
        messages = copy.deepcopy(messages)
        has_images = any(m.get("images") for m in messages)
        if has_images and self.processor is None:
            raise ValueError("Decider accepts text decisions only")
        if has_images:
            for message in messages:
                images = message.pop("images", [])
                message["content"] = [{"type": "text", "text": message.get("content", "")}] + [
                    {"type": "image", "url": "data:image/jpeg;base64," + image} for image in images]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=self.metadata["enable_thinking"], return_dict=True, return_tensors="pt")
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.metadata["enable_thinking"])
            inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if inputs["input_ids"].shape[-1] + max_tokens > self.metadata["max_seq_length"]:
            raise ValueError(f"{self.role} context exceeds its configured sequence limit; shorten the request/history")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                max_time=timeout, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        # Preserve tool-call/thinking delimiters even when a tokenizer marks them
        # special; strip only the actual generation terminator.
        text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        if self.tokenizer.eos_token:
            text = text.split(self.tokenizer.eos_token, 1)[0]
        return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--investigator-adapter", type=Path, required=True)
    p.add_argument("--decider-adapter", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    models = {"investigator": AdapterModel(args.investigator_adapter, args.load_in_4bit),
              "decider": AdapterModel(args.decider_adapter, args.load_in_4bit)}
    if any(model.role != key for key, model in models.items()):
        raise ValueError("Adapter roles do not match the supplied arguments")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path != "/api/chat":
                    raise ValueError("Expected /api/chat")
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                timeout = float(request.get("redence_timeout", 120))
                deadline = monotonic() + timeout
                if not lock.acquire(timeout=max(0, timeout)):
                    raise TimeoutError("Model server busy")
                try:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Request expired while waiting for model")
                    content = models[request["model"]].chat(request["messages"],
                        int(request.get("options", {}).get("num_predict", 2048)), remaining)
                finally:
                    lock.release()
                response = {"message": {"role": "assistant", "content": content}, "done": True}
                status = 200
            except Exception as exc:
                response, status = {"error": str(exc)}, 400
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f"Serving investigator and decider on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
