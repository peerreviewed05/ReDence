"""ReDence coordination, bounded decisions, trace capture, and report validation."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .protocol import decision_messages, parse_tool_call, strip_thinking

BROWSER_TOOLS = {"web_search", "fetch_page", "interact_page", "model_take_screenshot"}
VERDICTS = ["Benign", "Phishing", "Scam", "Malicious", "Insufficient evidence"]
DEFAULT_CATEGORIES = ["Fraudulent online store", "Investment or cryptocurrency scam",
    "Advance-fee scam", "Employment scam", "Fake giveaway/prize scam",
    "Technical-support scam", "Pet scam", "Other scam"]
SYSTEM = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")


class ModelClient:
    """Ollama-compatible transport; works with serve_models or exported Ollama models."""
    def __init__(self, host, name):
        self.host, self.name = host.rstrip("/"), name

    def chat(self, messages, timeout, max_tokens=2048):
        payload = {"model": self.name, "messages": messages, "stream": False,
            "think": self.name != "decider", "options": {"temperature": 0, "num_predict": max_tokens},
            "redence_timeout": max(0.01, timeout)}
        # The server uses adapter metadata for thinking; Ollama uses this explicit flag.
        if max_tokens <= 8:
            payload["think"] = False
        request = Request(self.host + "/api/chat", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=max(0.01, timeout)) as response:
                return json.load(response)["message"]["content"]
        except HTTPError as exc:
            raise ValueError(f"Model server: {exc.read().decode(errors='replace')}") from exc


class Controller:
    def __init__(self, investigator, decider, tools, output_dir, budget=120,
                 categories=None, clock=time.monotonic):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Budget must be positive and finite")
        self.investigator, self.decider, self.tools = investigator, decider, tools
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clock, self.deadline = clock, clock() + budget
        self.categories = categories or DEFAULT_CATEGORIES
        if (not isinstance(self.categories, list) or not self.categories
                or not all(isinstance(c, str) and c.strip() for c in self.categories)
                or len(set(self.categories)) != len(self.categories)):
            raise ValueError("Supply a nonempty list of unique category names")
        self.trace, self.ledger = [], []
        self.final_decisions = None

    def remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Investigation budget exhausted")
        return remaining

    def log(self, kind, **fields):
        item = {"trace_id": f"T{len(self.trace)+1}", "kind": kind, **fields}
        self.trace.append(item)
        with (self.output_dir / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item["trace_id"]

    @staticmethod
    def validate_operation(name, args):
        if name not in BROWSER_TOOLS or not isinstance(args, dict):
            raise ValueError("Operation must name an allowed browser tool with object arguments")
        allowed = {"fetch_page": {"url"}, "web_search": {"query"},
            "interact_page": {"url", "action", "id", "value", "key"},
            "model_take_screenshot": {"label"}}[name]
        if set(args) - allowed:
            raise ValueError("Unknown tool arguments")
        if name in {"fetch_page", "interact_page"}:
            u = urlparse(args.get("url", ""))
            if u.scheme not in {"http", "https"} or not u.netloc:
                raise ValueError("Tool URL must be HTTP or HTTPS")
        if name == "web_search" and not str(args.get("query", "")).strip():
            raise ValueError("Search query is required")
        if name == "interact_page":
            if args.get("action") not in {"click", "fill", "press"} or type(args.get("id")) is not int:
                raise ValueError("Interaction requires one action and a current integer mark id")
            if args["action"] == "fill" and not isinstance(args.get("value"), str):
                raise ValueError("Fill requires a string value")
            if args["action"] == "press" and not isinstance(args.get("key"), str):
                raise ValueError("Press requires a key")

    def execute(self, name, args):
        self.remaining()
        self.validate_operation(name, args)
        request_id = self.log("tool_request", tool=name, arguments=args)
        try:
            observation, image = self.tools.execute(name, args)
        except Exception as exc:
            observation, image = {"error": str(exc)}, None
        trace_id = self.log("observation", request_id=request_id, tool=name,
                            observation=observation)
        return {"trace_id": trace_id, "observation": observation}, image

    def delegate(self, state, question, options):
        messages = decision_messages(state, question, options)
        # Validate every operation before asking the model. It cannot invent arguments.
        for option in options:
            if "operation" in option:
                op = option["operation"]
                self.validate_operation(op["name"], op["arguments"])
        request_id = self.log("decision_request", state=state, question=question, options=options)
        answer = self.decider.chat(messages, self.remaining(), max_tokens=8).strip()
        if answer not in {o["label"] for o in options}:
            self.log("decision_error", request_id=request_id, raw_answer=answer)
            raise ValueError("Decider must return exactly one supplied option label")
        selected = next(o for o in options if o["label"] == answer)
        decision = {"label": answer, "key": selected["key"]}
        decision_id = self.log("judgment", request_id=request_id, decision=decision)
        result, image = {"decision": decision, "decision_trace_id": decision_id}, None
        if "operation" in selected:
            op = selected["operation"]
            observed, image = self.execute(op["name"], op["arguments"])
            result.update(observed)
        return result, image

    def record_evidence(self, **item):
        required = {"observation", "source_url", "evidence_type", "status", "relevance", "trace_ids"}
        if set(item) != required or not all(isinstance(item[k], str) and item[k].strip()
                                          for k in required - {"trace_ids"}):
            raise ValueError("Evidence needs observation, source_url, evidence_type, status, relevance, trace_ids")
        if item["status"] not in {"Direct observation", "Inference"}:
            raise ValueError("Invalid evidence status")
        if item["evidence_type"] not in {"Website content", "Website interaction",
                "Redirect or network behavior", "External source", "Technical metadata"}:
            raise ValueError("Invalid evidence type")
        observation_ids = {t["trace_id"] for t in self.trace if t["kind"] == "observation"}
        if not isinstance(item["trace_ids"], list) or not item["trace_ids"] or not set(item["trace_ids"]) <= observation_ids:
            raise ValueError("Evidence must cite existing observation trace IDs, not model judgments")
        cited = [t["observation"] for t in self.trace if t["trace_id"] in item["trace_ids"]]
        if item["source_url"] not in json.dumps(cited, ensure_ascii=False):
            raise ValueError("Evidence source URL must appear in the cited observations")
        item = {"id": f"E{len(self.ledger)+1}", **item}
        self.ledger.append(item)
        self.log("evidence", evidence=item)
        return {"evidence_id": item["id"]}

    def select_final(self, url):
        state = {"target_url": url, "current_goal": "Final website assessment",
                 "evidence_ledger": self.ledger}
        options = [{"label": chr(65+i), "key": v, "description": v} for i, v in enumerate(VERDICTS)]
        result, _ = self.delegate(state, "Which verdict is supported by the accumulated evidence?", options)
        verdict, category = result["decision"]["key"], None
        if verdict == "Scam":
            choices = self.categories
            # Hierarchical selection keeps each multiple-choice request within A..Z.
            # This also permits the paper's full BBB catalog via --categories-file.
            while len(choices) > 25:
                groups = [choices[i:i+25] for i in range(0, len(choices), 25)]
                if len(groups) > 25:
                    raise ValueError("Category catalog is too large")
                options = [{"label": chr(65+i), "key": str(i), "description": "; ".join(group)}
                           for i, group in enumerate(groups)]
                options.append({"label": chr(65+len(groups)), "key": "insufficient", "description": "Insufficient evidence"})
                result, _ = self.delegate(state, "Which category group contains the best supported category?", options)
                if result["decision"]["key"] == "insufficient":
                    choices = []
                    break
                choices = groups[int(result["decision"]["key"])]
            choices = choices + ["Insufficient evidence"]
            if len(choices) == 1:
                self.final_decisions = {"verdict": verdict, "category": "Insufficient evidence"}
                return self.final_decisions
            options = [{"label": chr(65+i), "key": v, "description": v} for i, v in enumerate(choices)]
            result, _ = self.delegate(state, "Which scam category is supported?", options)
            category = result["decision"]["key"]
        self.final_decisions = {"verdict": verdict, "category": category}
        return self.final_decisions

    def validate_report(self, report, url):
        import re
        required = {"url", "accessibility", "verdict", "category", "confidence", "summary", "explanation", "key_evidence"}
        if not isinstance(report, dict) or not required <= report.keys():
            raise ValueError("Report missing required fields")
        if report["url"] != url or report["accessibility"] not in {"Accessible", "Inaccessible"}:
            raise ValueError("Invalid URL or accessibility")
        if not self.final_decisions or any(report[k] != v for k, v in self.final_decisions.items()):
            raise ValueError("Report must match Decider's final verdict/category")
        confidence = report["confidence"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a number from 0 to 1")
        ids = {e["id"] for e in self.ledger}
        if not isinstance(report["key_evidence"], list):
            raise ValueError("key_evidence must be a list")
        for claim in report["key_evidence"]:
            if not isinstance(claim, dict) or not all(claim.get(k) for k in ("claim", "relevance", "evidence_ids")):
                raise ValueError("Every key claim needs evidence references and relevance")
            refs = claim["evidence_ids"]
            if not isinstance(refs, list) or not set(refs) <= ids:
                raise ValueError("Unknown Evidence ID in key_evidence")
        for field in ("summary", "explanation"):
            if not isinstance(report[field], str) or not report[field].strip():
                raise ValueError(f"Missing {field}")
            refs = set(re.findall(r"\[(E\d+)\]", report[field]))
            if not refs <= ids or (ids and not refs):
                raise ValueError(f"Invalid or missing Evidence IDs in {field}")
        if report["verdict"] != "Insufficient evidence" and (not ids or not report["key_evidence"]):
            raise ValueError("A substantive verdict requires recorded evidence")
        return {**report, "evidence_ledger": self.ledger, "status": "completed"}

    def run(self, url, domain_intel="", archive_url=None):
        self.url = url
        intel_trace_id = None
        if domain_intel.strip():
            intel_trace_id = self.log("observation", tool="precollected_domain_intel",
                observation={"url": url, "text": domain_intel})
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            f"Investigate {url}. Current date: {date.today()}. Time budget: {self.remaining():.1f}s. "
            f"Category alternatives: {json.dumps(self.categories)}. Pre-collected domain intelligence "
            f"(observation trace_id={intel_trace_id}): {domain_intel or 'Not supplied'}"}]
        initial, image = self.execute("fetch_page", {"url": url})
        if initial["observation"].get("error") and archive_url:
            initial, image = self.execute("fetch_page", {"url": archive_url})
        if initial["observation"].get("error"):
            return self.fallback(url, "Website and supplied archive inaccessible", "Inaccessible")
        messages.append({"role": "user", "content": json.dumps(initial, ensure_ascii=False)})
        for step in range(100):
            text = self.investigator.chat(messages, self.remaining())
            self.log("investigator", content=text)
            messages.append({"role": "assistant", "content": text})
            try:
                call = parse_tool_call(text)
                if call:
                    if self.final_decisions is not None:
                        raise ValueError("Final decisions selected; return the JSON report without more tools")
                    name, args = call
                    if name == "delegate_decision":
                        result, image = self.delegate(**args)
                    elif name == "record_evidence":
                        result, image = self.record_evidence(**args), None
                    else:
                        result, image = self.execute(name, args)
                    messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    if image:
                        messages.append({"role": "user", "content": "Current browser screenshot.", "images": [image]})
                else:
                    draft = strip_thinking(text)
                    if draft.startswith("```"):
                        draft = draft.split("\n", 1)[1].rsplit("```", 1)[0]
                    report = json.loads(draft)
                    if self.final_decisions is None:
                        if not self.ledger:
                            raise ValueError("Record evidence with record_evidence before the final report")
                        selected = self.select_final(url)
                        messages.append({"role": "user", "content":
                            "Final Decider selections: " + json.dumps(selected) +
                            ". Return the evidence-linked JSON report matching these selections."})
                    else:
                        return self.validate_report(report, url)
            except (ValueError, KeyError, TypeError) as exc:
                self.log("validation_error", error=str(exc))
                messages.append({"role": "user", "content": f"Controller validation error: {exc}. Correct your next response."})
        return self.fallback(url, "Maximum turn count reached")

    def fallback(self, url, reason, accessibility="Accessible"):
        return {"url": url, "accessibility": accessibility, "verdict": "Insufficient evidence",
            "category": None, "confidence": 0.0, "summary": reason, "explanation": reason,
            "key_evidence": [], "evidence_ledger": self.ledger, "status": "incomplete"}


def write_report(path, report):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv=None):
    from .config import load_env
    load_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--investigator-model", default=os.environ.get("REDENCE_INVESTIGATOR_MODEL", "investigator"))
    p.add_argument("--decider-model", default=os.environ.get("REDENCE_DECIDER_MODEL", "decider"))
    p.add_argument("--model-host", default=os.environ.get("REDENCE_MODEL_HOST", "http://127.0.0.1:11435"))
    p.add_argument("--budget", type=float, default=120)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/redence") / uuid.uuid4().hex[:12])
    p.add_argument("--categories-file", type=Path, help="JSON list of category alternatives (e.g. BBB categories)")
    p.add_argument("--domain-intel", type=Path, help="Previously collected WHOIS/DNS/SSL text")
    p.add_argument("--archive-url", help="HTTP(S) replay URL, used if the live site is inaccessible")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not math.isfinite(args.budget) or args.budget <= 0:
        p.error("--budget must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        if (args.output_dir / "trace.jsonl").exists() or (args.output_dir / "report.json").exists():
            p.error("Output directory already contains a run; choose a fresh --output-dir")
        # Isolate each website so a blocked model/browser call cannot exceed the budget.
        worker_args = list(argv if argv is not None else sys.argv[1:])
        if "--output-dir" not in worker_args:
            worker_args += ["--output-dir", str(args.output_dir)]
        cmd = [sys.executable, "-m", "redence.controller", *worker_args, "--worker"]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        child = subprocess.Popen(cmd, start_new_session=True, env=env)
        try:
            code = child.wait(timeout=args.budget)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            ledger = []
            trace_path = args.output_dir / "trace.jsonl"
            if trace_path.exists():
                for line in trace_path.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                        if entry["kind"] == "evidence":
                            ledger.append(entry["evidence"])
                    except (ValueError, KeyError):
                        pass
            report = {"url": args.url, "verdict": "Insufficient evidence", "category": None,
                "confidence": 0.0, "accessibility": "Unknown", "status": "budget_exhausted",
                "summary": "Investigation budget exhausted", "explanation": "Investigation budget exhausted",
                "key_evidence": [], "evidence_ledger": ledger}
            write_report(args.output_dir / "report.json", report)
            code = 2
        report_path = args.output_dir / "report.json"
        if not report_path.exists():
            write_report(report_path, {"url": args.url, "verdict": "Insufficient evidence",
                "status": "error", "error": f"Worker exited with code {code} before writing a report"})
        print(report_path)
        return code
    from .browser_tools import BrowserTools
    controller = None
    try:
        categories = json.loads(args.categories_file.read_text()) if args.categories_file else None
        with BrowserTools(args.output_dir) as tools:
            controller = Controller(ModelClient(args.model_host, args.investigator_model),
                ModelClient(args.model_host, args.decider_model), tools, args.output_dir,
                budget=args.budget, categories=categories)
            report = controller.run(args.url,
                args.domain_intel.read_text() if args.domain_intel else "",
                args.archive_url)
    except Exception as exc:
        if controller:
            report = controller.fallback(args.url, str(exc))
        else:
            report = {"url": args.url, "verdict": "Insufficient evidence", "status": "error", "error": str(exc)}
    write_report(args.output_dir / "report.json", report)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
