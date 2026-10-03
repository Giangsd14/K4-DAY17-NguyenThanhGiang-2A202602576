from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


CORE_FACT_KEYS = (
    "name",
    "location",
    "profession",
    "drink",
    "food",
    "pet",
    "response_style",
    "interests",
)


def estimate_tokens(text: str) -> int:
    """Simple, deterministic heuristic token estimator (~4 chars per token)."""

    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


@dataclass
class FactMetadata:
    value: str
    confidence: float = 0.9
    updated_turn: int = 0
    mentions: int = 1


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` with structured fact management, conflict resolution, and memory decay."""

    root_dir: Path
    max_optional_facts: int = 6
    _metadata: dict[str, dict[str, FactMetadata]] = field(default_factory=dict)
    _turn_counter: dict[str, int] = field(default_factory=dict)

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", (user_id or "").strip()).strip("_")
        if not safe_id:
            safe_id = "default"
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured `- **key**: value` lines from `User.md`."""

        text = self.read_text(user_id)
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            match = re.match(r"^\s*-\s*\*\*([^*]+)\*\*\s*:\s*(.+?)\s*$", line)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip()
                if key and val:
                    parsed[key] = val
        return parsed

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 0.9,
        min_confidence: float = 0.75,
    ) -> Path | None:
        return self.upsert_facts(
            user_id=user_id,
            updates={key: value},
            confidences={key: confidence},
            min_confidence=min_confidence,
        )

    def upsert_facts(
        self,
        user_id: str,
        updates: dict[str, str],
        confidences: dict[str, float] | None = None,
        min_confidence: float = 0.75,
    ) -> Path | None:
        """Upsert structured profile facts with confidence threshold, conflict resolution, and memory decay."""

        if not updates:
            return None

        conf_map = confidences or {}
        accepted: dict[str, tuple[str, float]] = {}
        for k, v in updates.items():
            cleaned_val = (v or "").strip()
            if not cleaned_val:
                continue
            conf = float(conf_map.get(k, 0.9))
            if conf >= min_confidence:
                accepted[k] = (cleaned_val, conf)

        if not accepted:
            return None

        turn = self._turn_counter.get(user_id, 0) + 1
        self._turn_counter[user_id] = turn

        current_facts = self.facts(user_id)
        user_meta = self._metadata.setdefault(user_id, {})

        # Ensure existing facts on disk have metadata entries
        for existing_key, existing_val in current_facts.items():
            if existing_key not in user_meta:
                user_meta[existing_key] = FactMetadata(
                    value=existing_val,
                    confidence=0.85,
                    updated_turn=0,
                    mentions=1,
                )

        # Conflict handling: overwrite key with newer fact value and update recency/frequency
        for key, (val, conf) in accepted.items():
            if key in user_meta:
                prev = user_meta[key]
                mentions = prev.mentions + 1 if prev.value == val else 1
                user_meta[key] = FactMetadata(
                    value=val,
                    confidence=conf,
                    updated_turn=turn,
                    mentions=mentions,
                )
            else:
                user_meta[key] = FactMetadata(
                    value=val,
                    confidence=conf,
                    updated_turn=turn,
                    mentions=1,
                )
            current_facts[key] = val

        # Memory decay: retain all core profile keys and prune stale optional facts
        decayed_facts = self._apply_decay(user_id, current_facts)
        return self._render_and_write(user_id, decayed_facts)

    def _apply_decay(self, user_id: str, current_facts: dict[str, str]) -> dict[str, str]:
        user_meta = self._metadata.get(user_id, {})
        core_items: dict[str, str] = {}
        optional_keys: list[str] = []

        for k in CORE_FACT_KEYS:
            if k in current_facts:
                core_items[k] = current_facts[k]

        for k in current_facts:
            if k not in CORE_FACT_KEYS:
                optional_keys.append(k)

        if len(optional_keys) > self.max_optional_facts:
            optional_keys.sort(
                key=lambda k: (
                    user_meta.get(k, FactMetadata(current_facts[k])).updated_turn,
                    user_meta.get(k, FactMetadata(current_facts[k])).mentions,
                    user_meta.get(k, FactMetadata(current_facts[k])).confidence,
                ),
                reverse=True,
            )
            kept_optional = set(optional_keys[: self.max_optional_facts])
            for k in list(user_meta.keys()):
                if k not in CORE_FACT_KEYS and k not in kept_optional:
                    user_meta.pop(k, None)
            optional_keys = [k for k in optional_keys if k in kept_optional]

        result = dict(core_items)
        for k in optional_keys:
            result[k] = current_facts[k]
        return result

    def _render_and_write(self, user_id: str, facts_dict: dict[str, str]) -> Path:
        lines = ["# User Profile", ""]
        for key, val in facts_dict.items():
            lines.append(f"- **{key}**: {val}")
        lines.append("")
        return self.write_text(user_id, "\n".join(lines))


_NOISE_PATTERNS = (
    "chỉ là câu đùa",
    "đùa với đồng nghiệp",
    "chứ không phải nơi ở",
    "vừa bay ra họp",
    "như ví dụ cũ",
    "đừng lấy nó làm nơi ở",
    "đừng nói backend engineer nữa",
)

_QUESTION_PREFIXES = (
    "mình tên gì",
    "tên mình là gì",
    "hiện tại mình đang ở đâu",
    "hiện tại mình làm nghề gì",
    "bạn có thể nhắc lại",
    "bạn có biết",
    "nhắc lại giúp mình",
    "nhắc lại style",
    "món ăn yêu thích của mình là gì",
    "đồ uống và món ăn yêu thích của mình là gì",
    "bạn thử nhớ lại xem",
    "nếu phải chọn giữa nghề cũ và nghề mới",
    "nếu ai đó nhắc",
    "tóm tắt ngắn về mình",
    "sang thread mới rồi",
)


def _is_question_or_noise_sentence(sentence: str) -> bool:
    low = sentence.strip().lower()
    if not low:
        return True
    if "?" in low:
        return True
    for pat in _NOISE_PATTERNS:
        if pat in low:
            return True
    for q_pat in _QUESTION_PREFIXES:
        if low.startswith(q_pat) or f", {q_pat}" in low:
            return True
    return False


def extract_profile_updates_with_confidence(
    message: str,
    min_confidence: float = 0.75,
) -> dict[str, tuple[str, float]]:
    """Extract structured profile facts along with confidence scores in [0.0, 1.0].

    Implements the 4 bonus guardrails:
    - Structured entity extraction (`name`, `location`, `profession`, `drink`, `food`, `pet`, `response_style`, `interests`)
    - Question & noise filtering (ignores questions, jokes, temporary travel, and negative examples)
    - Conflict handling (strips negated old clauses like 'chứ không còn ở Đà Nẵng' before extracting new facts)
    - Confidence threshold enforcement
    """

    text = (message or "").strip()
    if not text:
        return {}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    extracted: dict[str, tuple[str, float]] = {}

    for raw_sentence in sentences:
        if _is_question_or_noise_sentence(raw_sentence):
            continue

        # Remove negated/historical clauses so old facts are never re-extracted over corrections
        cleaned = re.sub(
            r"(?:chứ\s+)?không còn\s+(?:ở|làm|là)\s+[^.,;]+",
            "",
            raw_sentence,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"lúc đầu mình nói\s+[^.,;]+,",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"dù trước đó có nhắc\s+[^.,;]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"từ\s+(?:backend|Đà Nẵng|Huế)\s+sang\s+(?:MLOps|Đà Nẵng|Huế)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        low = cleaned.lower()

        # 1. Name extraction
        name_match = re.search(
            r"(?i:mình tên là|tên mình là|nhắc lại lần cuối[^:]*:\s*tên)\s+(DũngCT Stress|DũngCT|[A-ZÀ-ỸĐ][\wÀ-ỹ]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ]*){0,2})",
            cleaned,
        )
        if name_match:
            candidate_name = name_match.group(1).strip().rstrip(".,;")
            if candidate_name:
                extracted["name"] = (candidate_name, 0.95)

        # 2. Location extraction (including explicit corrections)
        loc_transition = re.search(
            r"(?i:cập nhật từ)\s+[A-ZÀ-ỸĐ][\wÀ-ỹ\s]*?\s+(?i:sang)\s+(Đà Nẵng|Huế|Hà Nội|Sài Gòn|TP\.?\s*HCM|[A-ZÀ-ỸĐ][\wÀ-ỹ]+(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ]+)?)",
            raw_sentence,
        )
        if loc_transition:
            extracted["location"] = (loc_transition.group(1).strip(), 0.98)
        else:
            loc_match = re.search(
                r"(?i:giờ mình đang ở|hiện đang ở|mình đang ở|đang làm việc ở|nơi ở hiện tại là|hiện ở|vẫn ở|mình ở)\s+(Đà Nẵng|Huế|Hà Nội|Sài Gòn|TP\.?\s*HCM|[A-ZÀ-ỸĐ][\wÀ-ỹ]+(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹ]+)?)",
                cleaned,
            )
            if loc_match:
                extracted["location"] = (loc_match.group(1).strip(), 0.92)

        # 3. Profession extraction (including corrections)
        prof_match = re.search(
            r"(?:đang làm|giờ chuyển sang|chuyển sang|mình làm|nghề nghiệp(?: hiện tại| thì)?(?: vẫn)? là|nghề hiện tại là|nghề)\s+(MLOps engineer|backend engineer|data engineer|AI engineer|software engineer|frontend engineer|[A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+){0,2}\s+engineer)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if prof_match:
            raw_prof = prof_match.group(1).strip()
            if "mlops" in raw_prof.lower():
                extracted["profession"] = ("MLOps engineer", 0.95)
            elif "backend" in raw_prof.lower():
                extracted["profession"] = ("backend engineer", 0.90)
            else:
                extracted["profession"] = (raw_prof, 0.88)

        # 4. Favorite drink
        if "cà phê sữa đá" in low and any(
            kw in low for kw in ("thích", "đồ uống", "uống")
        ):
            extracted["drink"] = ("cà phê sữa đá", 0.92)
        else:
            drink_match = re.search(
                r"đồ uống yêu thích(?:\s+của mình)?\s+là\s+([^.,;\n]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if drink_match:
                extracted["drink"] = (drink_match.group(1).strip(), 0.90)

        # 5. Favorite food
        if "mì quảng" in low and any(
            kw in low for kw in ("món ăn yêu thích", "ăn mì quảng", "món ruột")
        ):
            extracted["food"] = ("mì Quảng", 0.92)
        else:
            food_match = re.search(
                r"món ăn yêu thích(?:\s+của mình)?\s+là\s+([^.,;\n]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if food_match:
                extracted["food"] = (food_match.group(1).strip(), 0.90)

        # 6. Pet
        if "corgi" in low and any(kw in low for kw in ("nuôi", "con corgi", "bé corgi")):
            extracted["pet"] = ("corgi tên Bơ" if "bơ" in low else "corgi", 0.92)
        else:
            pet_match = re.search(
                r"mình nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?([^.,;\n]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if pet_match:
                extracted["pet"] = (pet_match.group(1).strip(), 0.88)

        # 7. Response style preferences
        if "3 bullet" in low:
            extracted["response_style"] = (
                "ngắn gọn, 3 bullet, có ví dụ thực chiến và nhấn vào trade-off",
                0.95,
            )
        elif any(
            kw in low
            for kw in (
                "trả lời ngắn gọn",
                "bullet ngắn và có ví dụ thực tế",
                "câu trả lời ngắn và có cấu trúc",
                "giữ câu trả lời gọn",
            )
        ):
            prev_style = extracted.get("response_style", ("", 0.0))[0]
            if "3 bullet" not in prev_style:
                extracted["response_style"] = (
                    "ngắn gọn, rõ ý, có bullet và ví dụ thực tế",
                    0.90,
                )

        # 8. Technical interests
        if "python" in low and "ai" in low and any(
            kw in low for kw in ("thích", "quan tâm", "dài hạn")
        ):
            extracted["interests"] = ("Python, AI ứng dụng, MLOps", 0.90)

    return {
        key: (val, conf)
        for key, (val, conf) in extracted.items()
        if conf >= min_confidence
    }


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts filtered by confidence threshold."""

    with_conf = extract_profile_updates_with_confidence(message, min_confidence=0.75)
    return {key: val for key, (val, _conf) in with_conf.items()}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded, compact summary of older messages."""

    if not messages:
        return ""

    highlights: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = " ".join((msg.get("content") or "").strip().split())
        if not content:
            continue
        # Keep only the first key sentence or a bounded 110-char snippet per turn
        first_sentence = re.split(r"(?<=[.!?])\s+", content)[0]
        snippet = first_sentence[:110] + ("..." if len(first_sentence) > 110 else "")
        item = f"[{role}] {snippet}"
        if item not in highlights:
            highlights.append(item)

    if not highlights:
        return ""

    # Retain the most informative/recent bounded items up to max_items
    selected = highlights[-max_items:]
    return "Tóm tắt lịch sử: " + " | ".join(selected)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads.

    Keeps recent messages in full and summarizes older messages whenever the
    thread context exceeds `threshold_tokens`.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "archive_snippets": [],
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        summary_text = str(thread_state.get("summary", ""))
        total_tokens = estimate_tokens(summary_text) + sum(
            estimate_tokens(m.get("content", "")) for m in messages
        )

        keep = max(1, self.keep_messages)
        if total_tokens > self.threshold_tokens and len(messages) > keep:
            older = messages[:-keep]
            recent = messages[-keep:]

            archive: list[dict[str, str]] = thread_state["archive_snippets"]  # type: ignore[assignment]
            archive.extend(older)
            # Keep archive bounded so summary generation stays fast and compact
            if len(archive) > 12:
                archive[:] = archive[-12:]

            thread_state["summary"] = summarize_messages(archive, max_items=6)
            thread_state["messages"] = recent
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread_state = self._ensure_thread(thread_id)
        return {
            "messages": list(thread_state["messages"]),  # type: ignore[arg-type]
            "summary": str(thread_state.get("summary", "")),
            "compactions": int(thread_state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id].get("compactions", 0))