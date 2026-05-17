#!/usr/bin/env python3
"""Generate synthetic coding-agent and RAG-agent workload JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


TOKEN_LENGTHS = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
}

LAYOUTS = ("front_volatile", "middle_volatile", "end_volatile", "stable_prefix")
WORKLOADS = ("coding", "rag")


CODING_TASKS = [
    "Explain the scheduler module and list the key execution paths.",
    "Find the most likely bug in the cache eviction code.",
    "Propose a minimal fix for the failing cache reuse test.",
    "Patch the token accounting helper to handle empty completions.",
    "Add focused unit tests for the benchmark CSV writer.",
    "Use this test failure to refine the latency aggregation logic.",
    "Summarize the changes and call out remaining risks.",
    "Review the CLI flags for confusing defaults.",
    "Refactor the workload generator without changing output shape.",
    "Add validation for unsupported prompt layouts.",
    "Explain why warm-cache turn latency differs from cold-cache latency.",
    "Identify missing instrumentation around server memory usage.",
    "Improve error reporting for interrupted streaming responses.",
    "Add a small smoke-test command to the benchmark suite.",
    "Check whether the result schema is stable enough for plotting.",
    "Suggest a safer restart boundary between llama-server configs.",
    "Write a concise note for reproducing the 8K experiment.",
    "Inspect the generated prompt and identify volatile prefix data.",
    "Make the prompt layout more cache-friendly.",
    "Prepare the final benchmark checklist.",
]

RAG_QUESTIONS = [
    "Summarize the document's main claim in three bullet points.",
    "What definition is given for time to first token?",
    "Compare the no-cache and stable-prefix strategies.",
    "Which configuration is safest for a 32 GB Apple Silicon laptop?",
    "What risks can make a 32K context experiment noisy?",
    "Which metric best captures prefill cost?",
    "Why is a changing timestamp at the beginning of the prompt harmful?",
    "What should be reported for warm-cache turns?",
    "Which result would invalidate the cache-reuse hypothesis?",
    "How should repeated runs be summarized?",
    "What does the document say about single concurrency?",
    "Which plots are required for the core result?",
    "What should happen if server RSS exceeds the safety threshold?",
    "How does output length affect total latency speedup?",
    "Which experiment isolates prompt layout sensitivity?",
    "What is the recommended minimum deliverable?",
    "Which model size should be used first?",
    "How should the model path be configured?",
    "What data should be saved in the raw CSV?",
    "List the final reproduction steps.",
]


def approx_tokens(text: str) -> int:
    ascii_chars = 0
    non_ascii_chars = 0
    for char in text:
        if char.isspace():
            continue
        if ord(char) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    return max(1, math.ceil(ascii_chars / 4 + non_ascii_chars / 1.8))


def token_label(value: int) -> str:
    for label, tokens in TOKEN_LENGTHS.items():
        if value == tokens:
            return label
    return str(value)


def fixed_metadata(turn_id: int) -> str:
    timestamp = datetime(2026, 5, 12, 21, 0, 0) + timedelta(seconds=turn_id * 37)
    request_id = uuid.uuid5(uuid.NAMESPACE_URL, f"kv-cache-benchmark-{turn_id}")
    return (
        "[Dynamic Metadata]\n"
        f"current_time = {timestamp.isoformat()}\n"
        f"request_id = {request_id}\n"
        f"turn_id = {turn_id}\n"
        "latest_error = synthetic benchmark status line changed for this turn\n"
    )


def coding_stable_base() -> str:
    return """[System Prompt]
You are a local coding assistant. Be precise, concise, and base every answer on
the provided repository context.

[Tool Definitions]
read_file(path): return file contents.
grep_search(query): search repository text.
edit_file(path, patch): apply a patch.
run_tests(command): run a local test command.

[Repository Summary]
The synthetic project benchmarks local LLM agent prompt caching. It contains a
workload generator, a streaming benchmark client, a suite runner that restarts
llama-server between cache strategies, and plotting scripts that summarize warm
and cold turns.

[Relevant Files]
"""


def coding_filler_block(index: int) -> str:
    return f"""
--- file: benchmark/cache_policy_{index}.py
class CachePolicy{index}:
    def __init__(self, reuse_tokens: int, layout: str) -> None:
        self.reuse_tokens = reuse_tokens
        self.layout = layout
        self.events = []

    def record_prefill(self, prompt_tokens: int, reused_tokens: int) -> dict:
        missed_tokens = max(0, prompt_tokens - reused_tokens)
        event = {{
            "prompt_tokens": prompt_tokens,
            "reused_tokens": reused_tokens,
            "missed_tokens": missed_tokens,
            "layout": self.layout,
        }}
        self.events.append(event)
        return event

def explain_turn_{index}(turn_id: int) -> str:
    if turn_id <= 1:
        return "cold-cache turn"
    return "warm-cache turn with reusable stable prefix"
"""


def rag_stable_base() -> str:
    return """[System Prompt]
You are a careful document QA assistant. Answer only from the provided document.
If the document does not contain enough evidence, say that the answer is not
available from the document.

[Document Context]
"""


def rag_filler_block(index: int) -> str:
    return f"""
Section {index}: Local agent cache benchmark notes.
Prompt caching reduces repeated prefill work when consecutive requests share a
stable prefix. The benchmark records time to first token, total latency, decode
throughput, prompt length, output length, and server memory. The strongest
controlled comparison keeps the same semantic content while moving volatile
metadata between the beginning, middle, and end of the prompt. On a 32 GB Apple
Silicon laptop, 2K through 16K prompt lengths are the main matrix; 32K is an
optional stress test because memory pressure and thermal throttling can add
noise. Single concurrency is preferred for the first pass so slot scheduling
does not hide the prompt-cache effect.
"""


def build_stable_context(workload: str, target_tokens: int) -> str:
    if workload == "coding":
        text = coding_stable_base()
        block_fn = coding_filler_block
    else:
        text = rag_stable_base()
        block_fn = rag_filler_block

    # Leave room for role wrappers, volatile metadata, and the user request.
    stable_target = max(256, target_tokens - 350)
    index = 1
    while approx_tokens(text) < stable_target:
        text += block_fn(index)
        index += 1
    return text


def turn_text(workload: str, turn_id: int) -> str:
    if workload == "coding":
        return CODING_TASKS[(turn_id - 1) % len(CODING_TASKS)]
    return RAG_QUESTIONS[(turn_id - 1) % len(RAG_QUESTIONS)]


def build_messages(workload: str, layout: str, stable_context: str, turn_id: int) -> list[dict]:
    metadata = fixed_metadata(turn_id)
    user_request = (
        "[Current User Request]\n"
        if workload == "coding"
        else "[Question]\n"
    ) + turn_text(workload, turn_id)

    if layout == "front_volatile":
        system = metadata + "\n" + stable_context
        user = user_request
    elif layout == "middle_volatile":
        midpoint = len(stable_context) // 2
        system = stable_context[:midpoint] + "\n" + metadata + "\n" + stable_context[midpoint:]
        user = user_request
    elif layout == "end_volatile":
        system = stable_context
        user = metadata + "\n" + user_request
    elif layout == "stable_prefix":
        system = stable_context
        user = metadata + "\n" + user_request
    else:
        raise ValueError(f"unsupported layout: {layout}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_lengths(values: list[str]) -> list[int]:
    lengths = []
    for value in values:
        normalized = value.lower()
        if normalized in TOKEN_LENGTHS:
            lengths.append(TOKEN_LENGTHS[normalized])
        else:
            lengths.append(int(value))
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="workloads")
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=WORKLOADS)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    parser.add_argument("--lengths", nargs="+", default=["2k", "4k"])
    parser.add_argument("--turns", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for workload in args.workloads:
        for target_tokens in parse_lengths(args.lengths):
            stable_context = build_stable_context(workload, target_tokens)
            for layout in args.layouts:
                records = []
                for turn_id in range(1, args.turns + 1):
                    messages = build_messages(workload, layout, stable_context, turn_id)
                    content = "\n".join(message["content"] for message in messages)
                    records.append(
                        {
                            "workload": workload,
                            "prompt_len_target": target_tokens,
                            "layout": layout,
                            "turn_id": turn_id,
                            "approx_prompt_tokens": approx_tokens(content),
                            "messages": messages,
                        }
                    )
                path = output_dir / f"{workload}_{token_label(target_tokens)}_{layout}.jsonl"
                count = write_jsonl(path, records)
                total += count
                print(f"wrote {count:3d} records -> {path}")
    print(f"wrote {total} records total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
