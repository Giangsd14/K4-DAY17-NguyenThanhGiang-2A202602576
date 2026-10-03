from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0.0 / 0.5 / 1.0 depending on how many expected facts appear in the answer."""

    if not expected:
        return 1.0
    ans_low = (answer or "").lower()
    hits = sum(1 for item in expected if item.lower() in ans_low)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Compute a deterministic quality score in [0.0, 1.0] for offline mode."""

    cleaned = (answer or "").strip()
    if not cleaned:
        return 0.0

    recall = recall_points(cleaned, expected)
    structure_bonus = 0.10 if ("- " in cleaned or "\n" in cleaned) else 0.03
    concise_bonus = 0.08 if 20 <= len(cleaned) <= 500 else 0.02

    if recall == 0.0:
        return round(0.15 + concise_bonus, 2)

    score = 0.78 * recall + structure_bonus + concise_bonus
    return round(min(1.0, score), 2)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: Any,
) -> BenchmarkRow:
    """Evaluate one agent over a list of conversations and cross-session recall questions."""

    users = {str(conv.get("user_id", "default")) for conv in conversations}
    if hasattr(agent, "profile_store"):
        for uid in users:
            profile_path = agent.profile_store.path_for(uid)
            if profile_path.exists():
                profile_path.unlink()
            agent.profile_store._metadata.pop(uid, None)
            agent.profile_store._turn_counter.pop(uid, None)

    initial_sizes = {
        uid: (agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0)
        for uid in users
    }

    all_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        thread_id = str(conv["id"])
        user_id = str(conv["user_id"])
        all_threads.append(thread_id)

        for turn in conv.get("turns", []):
            agent.reply(user_id, thread_id, str(turn))

        for idx, rq in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"{thread_id}-recall-{idx}"
            all_threads.append(recall_thread_id)
            question = str(rq["question"])
            expected = [str(x) for x in rq.get("expected_contains", [])]

            result = agent.reply(user_id, recall_thread_id, question)
            answer = str(result.get("response", ""))

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    total_agent_tokens = sum(agent.token_usage(tid) for tid in all_threads)
    total_prompt_tokens = sum(agent.prompt_token_usage(tid) for tid in all_threads)
    total_compactions = sum(agent.compaction_count(tid) for tid in all_threads)

    avg_recall = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    final_sizes = {
        uid: (agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0)
        for uid in users
    }
    memory_growth = sum(max(0, final_sizes[u] - initial_sizes[u]) for u in users)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a Markdown comparison table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    try:
        from tabulate import tabulate

        return str(tabulate(table_data, headers=headers, tablefmt="github"))
    except ImportError:
        widths = [
            max(len(str(headers[i])), *(len(str(row[i])) for row in table_data))
            for i in range(len(headers))
        ]
        header_line = (
            "| "
            + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )
        sep_line = (
            "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        )
        body_lines = [
            "| "
            + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
            + " |"
            for row in table_data
        ]
        return "\n".join([header_line, sep_line, *body_lines])


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""

    config = load_config(Path(__file__).resolve().parent.parent)

    standard_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")

    baseline_standard = BaselineAgent(config=config, force_offline=True)
    advanced_standard = AdvancedAgent(config=config, force_offline=True)

    standard_rows = [
        run_agent_benchmark("Baseline", baseline_standard, standard_convs, config),
        run_agent_benchmark("Advanced", advanced_standard, standard_convs, config),
    ]

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]

    print("=== Standard Benchmark (data/conversations.json) ===")
    print(format_rows(standard_rows))
    print()
    print("=== Long-Context Stress Benchmark (data/advanced_long_context.json) ===")
    print(format_rows(stress_rows))


if __name__ == "__main__":
    main()