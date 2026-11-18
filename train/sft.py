"""Completion-only QLoRA training shared by the two ReDence models."""
import argparse
import json
import os
from pathlib import Path

from train.data import (ROOT, DECIDER_DATA, INVESTIGATOR_DATA, load_bundle, load_json,
                        decider_rows, investigator_rows, summarize, tokenize_row)


def parse_args(role, argv=None):
    decider = role == "decider"
    p = argparse.ArgumentParser(description=f"Train the ReDence {role} adapter")
    p.add_argument("--model_name", default=("togethercomputer/Tev1-4B-experimental" if decider else "Qwen/Qwen3.5-4B"))
    p.add_argument("--data_file", type=Path, default=DECIDER_DATA if decider else INVESTIGATOR_DATA)
    p.add_argument("--decider_data_file", type=Path, default=DECIDER_DATA)
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / role)
    p.add_argument("--lora_dir", type=Path, default=ROOT / "models" / role)
    p.add_argument("--max_seq_length", type=int, default=2048 if decider else 16384)
    p.add_argument("--lora_r", type=int, default=8 if decider else 16)
    p.add_argument("--lora_alpha", type=int, default=16 if decider else 32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=1 if decider else 3)
    p.add_argument("--learning_rate", type=float, default=5e-5 if decider else 1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weighted_sampling", action=argparse.BooleanOptionalAction, default=decider)
    p.add_argument("--context_mode", choices=["full", "short", "short_tool_full_final"], default="full")
    p.add_argument("--tool_history_exchanges", type=int, default=2)
    p.add_argument("--max_examples", type=int, help="Debug cap on TRAINING rows only")
    p.add_argument("--validate_only", action="store_true", help="Check data/splits without ML dependencies")
    p.add_argument("--evaluate_test", action="store_true", help="Evaluate untouched test partition after training")
    args = p.parse_args(argv)
    if args.max_seq_length < 2 or (args.max_examples is not None and args.max_examples <= 0):
        p.error("Sequence length must be >=2 and max_examples must be positive")
    return args


def main(role, argv=None):
    args = parse_args(role, argv)
    bundle = load_bundle(args.data_file if role == "decider" else args.decider_data_file)
    if role == "decider":
        rows = decider_rows(bundle)
    else:
        rows, delegated = investigator_rows(load_json(args.data_file), bundle,
            args.context_mode, args.tool_history_exchanges)
        print(f"Replaced {delegated} recorded source selections with controller delegations")
    if args.max_examples:
        rows["train"] = rows["train"][:args.max_examples]
    print(json.dumps(summarize(rows), indent=2))
    if args.validate_only:
        return

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU; use --validate_only for a CPU data check")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    prepared, weights, skipped = {}, [], {}
    for split, items in rows.items():
        encoded, skipped[split] = [], 0
        for row in items:
            item = tokenize_row(row, tokenizer, args.max_seq_length, thinking=role == "investigator")
            if item is None:
                skipped[split] += 1
                continue
            encoded.append(item)
            if split == "train":
                weights.append(row["weight"])
        prepared[split] = Dataset.from_list(encoded) if encoded else None
    print(f"Overlength rows excluded (no token truncation): {skipped}")
    if prepared["train"] is None:
        raise ValueError("No training examples fit the chosen sequence length")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if args.load_in_4bit else None
    model = AutoModelForImageTextToText.from_pretrained(args.model_name,
        quantization_config=quantization, torch_dtype=dtype, device_map={"": torch.cuda.current_device()})
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))

    class WeightedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            if not args.weighted_sampling:
                return super()._get_train_sampler(train_dataset)
            if self.args.world_size != 1:
                raise ValueError("Weighted sampling currently supports one GPU; use --no-weighted_sampling for DDP")
            generator = torch.Generator().manual_seed(args.seed)
            return torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True,
                                                          generator=generator)

    training_args = TrainingArguments(output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, logging_steps=args.logging_steps,
        eval_strategy="steps" if prepared["validation"] is not None else "no",
        eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=2,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", seed=args.seed, label_names=["labels"])
    trainer = WeightedTrainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=prepared["train"], eval_dataset=prepared["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(args.lora_dir)
    tokenizer.save_pretrained(args.lora_dir)
    metadata = {"role": role, "base_model": args.model_name, "max_seq_length": args.max_seq_length,
                "enable_thinking": role == "investigator", "split_counts": summarize(rows),
                "overlength_excluded": skipped, "weighted_sampling": args.weighted_sampling}
    (args.lora_dir / "redence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.evaluate_test and prepared["test"] is not None:
        metrics = trainer.evaluate(prepared["test"], metric_key_prefix="test")
        (args.lora_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {role} adapter to {args.lora_dir}")
