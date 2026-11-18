"""Investigator trajectory expansion and compact context helpers."""
import random
import re
from collections import defaultdict
from typing import Any

_TOOL_RESPONSE_RE = re.compile(
    r"^<tool_response[^>]*>\s*(.*?)\s*</tool_response>\s*$",
    re.DOTALL,
)


ASSISTANT_HEADER = "<|im_start|>assistant\n"
TOOL_HISTORY_SKELETON_HEADER = (
    "Investigation tool history (compact, full run). "
    "Recent turns with full tool results follow when present."
)


def split_examples_by_run(
    examples: list[dict],
    test_size: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Keep all turns from a run_id in the same split (no cross-split leakage)."""
    by_run: dict[Any, list[dict]] = defaultdict(list)
    for ex in examples:
        by_run[ex["run_id"]].append(ex)

    run_ids = sorted(by_run.keys(), key=str)
    if not run_ids:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(run_ids)

    if test_size <= 0 or len(run_ids) == 1:
        n_test = 0
    else:
        n_test = int(round(len(run_ids) * test_size))
        n_test = min(max(n_test, 1), len(run_ids) - 1)

    test_ids = set(run_ids[:n_test])
    train, test = [], []
    for rid in run_ids:
        bucket = test if rid in test_ids else train
        bucket.extend(by_run[rid])
    return train, test


def unwrap_tool_content(content: str) -> str:
    """Qwen chat template already wraps tool msgs in <tool_response>; avoid double wrap."""
    if not isinstance(content, str):
        return content
    m = _TOOL_RESPONSE_RE.match(content.strip())
    return m.group(1) if m else content


def normalize_message(msg: dict) -> dict:
    role = msg["role"]
    content = msg.get("content", "")
    if role == "tool":
        return {"role": "tool", "content": unwrap_tool_content(content)}
    return {"role": role, "content": content}


def split_bootstrap(messages: list[dict]) -> tuple[list[dict], int]:
    """system + first user as fixed prefix; return (bootstrap, index_after)."""
    i = 0
    bootstrap: list[dict] = []

    if i < len(messages) and messages[i]["role"] == "system":
        bootstrap.append(messages[i])
        i += 1

    if i < len(messages) and messages[i]["role"] == "user":
        bootstrap.append(messages[i])
        i += 1

    return bootstrap, i


_TOOL_FN_RE = re.compile(r"<function=([^>\s]+)")
_TOOL_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def summarize_tool_call(content: str) -> str | None:
    """One-line compact summary of a Qwen XML tool_call, or None if absent."""
    if "<tool_call>" not in content:
        return None

    fn_m = _TOOL_FN_RE.search(content)
    name = fn_m.group(1) if fn_m else "unknown"

    parts: list[str] = []

    for key, val in _TOOL_PARAM_RE.findall(content):
        v = " ".join(val.split())

        if len(v) > 80:
            v = v[:77] + "..."

        parts.append(f"{key}={v}")

    return f"{name}({', '.join(parts)})" if parts else f"{name}()"


def build_compact_tool_history_message(prior: list[dict]) -> dict | None:
    """
    Synthetic user message listing all prior tool calls (no tool payloads).

    Used on both tool-call and final turns as a cheap global investigation
    skeleton. Recent exchanges are still included separately with full
    tool results.
    """
    lines: list[str] = []
    step = 0

    for m in prior:
        if m.get("role") != "assistant":
            continue

        summary = summarize_tool_call(m.get("content") or "")

        if not summary:
            continue

        step += 1
        lines.append(f"{step}. {summary}")

    if not lines:
        return None

    body = TOOL_HISTORY_SKELETON_HEADER + "\n" + "\n".join(lines)

    return {
        "role": "user",
        "content": body,
    }


def last_k_exchanges(
    prior: list[dict],
    tool_history_exchanges: int,
) -> list[dict]:
    """Keep the last K assistant(+following tool) exchanges from prior."""
    if tool_history_exchanges <= 0 or not prior:
        return []

    asst_idxs = [
        j for j, m in enumerate(prior)
        if m["role"] == "assistant"
    ]

    if not asst_idxs:
        return list(prior)

    start = (
        asst_idxs[-tool_history_exchanges]
        if len(asst_idxs) >= tool_history_exchanges
        else asst_idxs[0]
    )

    return prior[start:]


def build_turn_prompt(
    messages: list[dict],
    turn_index: int,
    turn_kind: str,
    context_mode: str,
    tool_history_exchanges: int,
) -> list[dict]:
    """
    Build prompt messages for one assistant turn.

    In short modes, BOTH tool-call and final turns receive:

      1. system + first user bootstrap
      2. compact skeleton containing all prior tool calls
      3. last K full assistant(+tool) exchanges

    This means older tool calls remain visible through the skeleton while
    recent exchanges retain their complete tool results.
    """
    full_prior = messages[:turn_index]

    if context_mode == "full":
        return full_prior

    # short / short_tool_full_final
    bootstrap, rest_start = split_bootstrap(full_prior)
    prior = full_prior[rest_start:]

    # Full recent context.
    window = last_k_exchanges(
        prior,
        tool_history_exchanges,
    )

    # Global compact history of all previous tool calls.
    skeleton = build_compact_tool_history_message(prior)

    out = list(bootstrap)

    if skeleton is not None:
        out.append(skeleton)

    out.extend(window)

    return out


def expand_trajectories(
    raw_records: list,
    context_mode: str = "short",
    tool_history_exchanges: int = 2,
    include_final_turns: bool = True,
) -> list[dict]:
    """Expand each run into one SFT example per assistant turn."""
    examples = []

    for rec in raw_records:
        messages = [
            normalize_message(m)
            for m in rec.get("messages", [])
        ]

        if not messages:
            continue

        run_id = rec.get("id", "unknown")

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            kind = (
                "tool"
                if "<tool_call>" in msg.get("content", "")
                else "final"
            )

            if kind == "final" and not include_final_turns:
                continue

            prompt = build_turn_prompt(
                messages,
                i,
                kind,
                context_mode=context_mode,
                tool_history_exchanges=tool_history_exchanges,
            )

            if not prompt:
                continue

            examples.append(
                {
                    "messages_prompt": prompt,
                    "messages_completion": [msg],
                    "run_id": run_id,
                    "turn_index": i,
                    "turn_kind": kind,
                }
            )

    return examples

