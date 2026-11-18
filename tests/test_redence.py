import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from redence.controller import Controller
from redence.protocol import decision_messages, encode_tool_call, parse_tool_call
from train.data import (load_bundle, decider_rows, investigator_rows, load_json,
                        INVESTIGATOR_DATA, tokenize_row)
from train.sft import parse_args


class FakeModel:
    def __init__(self, answers):
        self.answers, self.calls = iter(answers), []

    def chat(self, messages, timeout, max_tokens=2048):
        self.calls.append((messages, timeout, max_tokens))
        return next(self.answers)


class FakeTools:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return ({"error": "unavailable"} if self.fail else
                {"url": arguments.get("url"), "text": "Observed investment offer"}), None


def evidence(trace="T2"):
    return {"observation": "Investment offer", "source_url": "https://example.org",
            "evidence_type": "Website content", "status": "Direct observation",
            "relevance": "Supports stated purpose", "trace_ids": [trace]}


def options():
    return [{"label": "A", "key": "true", "description": "Yes", "operation": {
                "name": "fetch_page", "arguments": {"url": "https://example.org/about"}}},
            {"label": "B", "key": "false", "description": "No"}]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def controller(self, answers=(), investigator=(), tools=None, **kwargs):
        return Controller(FakeModel(investigator), FakeModel(answers), tools or FakeTools(),
                          self.directory.name, **kwargs)

    def test_selected_operation_runs_once_and_judgments_are_separate(self):
        c = self.controller(["A"])
        result, _ = c.delegate({"target_url": "https://example.org"}, "Useful source?", options())
        self.assertEqual(len(c.tools.calls), 1)
        self.assertEqual(result["decision"]["key"], "true")
        self.assertEqual([x["kind"] for x in c.trace],
                         ["decision_request", "judgment", "tool_request", "observation"])
        c.record_evidence(**evidence("T4"))
        with self.assertRaises(ValueError):
            c.record_evidence(**evidence("T2"))  # A model judgment is not an observation.

    def test_rejected_option_does_not_execute(self):
        c = self.controller(["B"])
        result, _ = c.delegate({}, "Useful source?", options())
        self.assertNotIn("observation", result)
        self.assertEqual(c.tools.calls, [])

    def test_invalid_decider_output_never_executes(self):
        for answer in ("C", "A because yes", "<think>yes</think>A", "0.9"):
            with self.subTest(answer=answer):
                c = self.controller([answer])
                with self.assertRaises(ValueError):
                    c.delegate({}, "Question?", options())
                self.assertEqual(c.tools.calls, [])

    def test_invalid_operation_rejected_before_model(self):
        c = self.controller(["A"])
        choices = options()
        choices[1]["operation"] = {"name": "shell", "arguments": {}}
        with self.assertRaises(ValueError):
            c.delegate({}, "Question?", choices)
        self.assertEqual(c.decider.calls, [])

    def test_full_investigation_and_final_delegation(self):
        report = {"url": "https://example.org", "accessibility": "Accessible", "verdict": "Scam",
            "category": "Investment or cryptocurrency scam", "confidence": 0.8,
            "summary": "Investment offer [E1]", "explanation": "Observed offer [E1]",
            "key_evidence": [{"claim": "Investment offer", "evidence_ids": ["E1"], "relevance": "Purpose"}]}
        c = self.controller(["C", "B"], [encode_tool_call("record_evidence", evidence()),
            json.dumps(report), json.dumps(report)])
        result = c.run("https://example.org")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evidence_ledger"][0]["id"], "E1")
        self.assertEqual(len(c.decider.calls), 2)
        self.assertEqual(len(c.tools.calls), 1)
        bad = {**report, "summary": "Invented [E999]"}
        with self.assertRaises(ValueError):
            c.validate_report(bad, "https://example.org")
        with self.assertRaises(ValueError):
            c.validate_report({**report, "verdict": "Benign"}, "https://example.org")

    def test_inaccessible_site_falls_back_without_calling_models(self):
        c = self.controller(tools=FakeTools(fail=True))
        report = c.run("https://example.org", archive_url="https://archive.example.org")
        self.assertEqual(report["accessibility"], "Inaccessible")
        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertEqual(len(c.tools.calls), 2)
        self.assertEqual(c.investigator.calls, [])

    def test_category_catalog_larger_than_alphabet(self):
        c = self.controller(["C", "B", "A"], categories=[f"Type {i}" for i in range(31)])
        final = c.select_final("https://example.org")
        self.assertEqual(final, {"verdict": "Scam", "category": "Type 25"})

    def test_expired_budget_prevents_tools(self):
        clock = iter([0, 121])
        c = self.controller(clock=lambda: next(clock))
        with self.assertRaises(TimeoutError):
            c.execute("fetch_page", {"url": "https://example.org"})
        self.assertEqual(c.tools.calls, [])

    def test_supervisor_writes_timeout_report(self):
        result = subprocess.run([sys.executable, "-m", "redence.controller", "--url",
            "https://example.org", "--budget", "0.00001", "--output-dir", self.directory.name],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads((Path(self.directory.name) / "report.json").read_text())
        self.assertEqual(report["status"], "budget_exhausted")


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_bundle_counts_weights_and_prompt_parity(self):
        rows = decider_rows(self.bundle)
        self.assertEqual([len(rows[s]) for s in ("train", "validation", "test")], [4269, 559, 577])
        self.assertTrue(all(r["weight"] == 1 for s in ("validation", "test") for r in rows[s]))
        for example in self.bundle["examples"]:
            user = json.loads(example["messages"][1]["content"])
            self.assertEqual(decision_messages(**user), example["messages"][:-1])

    def test_investigator_delegations_and_shared_domain_splits(self):
        rows, n = investigator_rows(load_json(INVESTIGATOR_DATA), self.bundle)
        self.assertEqual(n, 1684)
        self.assertEqual([len(rows[s]) for s in ("train", "validation", "test")], [18994, 2389, 2498])
        splits = {r["trajectory_id"]: r["split"] for r in self.bundle["trajectory_manifest"]}
        for split, items in rows.items():
            self.assertTrue(all(splits[r["id"].rsplit(":", 1)[0]] == split for r in items))
        sample = next(r for r in rows["train"] if "<function=delegate_decision>" in r["messages"][-1]["content"])
        name, args = parse_tool_call(sample["messages"][-1]["content"])
        self.assertEqual(name, "delegate_decision")
        self.assertIn("operation", args["options"][0])

    def test_role_defaults_match_paper(self):
        d, i = parse_args("decider", []), parse_args("investigator", [])
        self.assertEqual(d.model_name, "togethercomputer/Tev1-4B-experimental")
        self.assertEqual((d.max_seq_length, d.lora_r, d.lora_alpha, d.epochs, d.learning_rate),
                         (2048, 8, 16, 1, 5e-5))
        self.assertEqual((i.model_name, i.max_seq_length, i.lora_r, i.lora_alpha, i.epochs),
                         ("Qwen/Qwen3.5-4B", 16384, 16, 32, 3))

    def test_answer_and_eos_only_loss_no_silent_truncation(self):
        class Tokenizer:
            eos_token = "<EOS>"
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages) + "<assistant><think>\n\n</think>\n\n"
            def encode(self, text, **kwargs):
                return list(text.encode())
        row = decider_rows(self.bundle)["train"][0]
        item = tokenize_row(row, Tokenizer(), 10000)
        target = bytes(t for t in item["labels"] if t != -100).decode()
        self.assertEqual(target, "A<EOS>")
        self.assertTrue(all(t == -100 for t in item["labels"][:-len(target)]))
        self.assertIsNone(tokenize_row(row, Tokenizer(), 2))

    def test_xml_and_json_tool_calls(self):
        args = {"state": {"goal": "verify"}, "question": "Useful?", "options": options()}
        self.assertEqual(parse_tool_call(encode_tool_call("delegate_decision", args)), ("delegate_decision", args))
        self.assertEqual(parse_tool_call('<tool_call>{"name":"fetch_page","arguments":{"url":"https://example.org"}}</tool_call>'),
                         ("fetch_page", {"url": "https://example.org"}))
        with self.assertRaises(ValueError):
            parse_tool_call(encode_tool_call("fetch_page", {}) * 2)


if __name__ == "__main__":
    unittest.main()
