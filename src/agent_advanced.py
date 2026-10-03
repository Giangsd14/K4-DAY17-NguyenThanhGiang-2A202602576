from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    extract_profile_updates_with_confidence,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B (Advanced Agent): combines short-term memory, persistent `User.md`, and compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live LangChain mode and deterministic offline mode."""

        if not self.force_offline and self.langchain_agent is not None:
            try:
                min_conf = getattr(self.config, "confidence_threshold", 0.75)
                updates_with_conf = extract_profile_updates_with_confidence(
                    message, min_confidence=min_conf
                )
                if updates_with_conf:
                    updates = {k: v for k, (v, _c) in updates_with_conf.items()}
                    confs = {k: c for k, (_v, c) in updates_with_conf.items()}
                    self.profile_store.upsert_facts(
                        user_id, updates, confidences=confs, min_confidence=min_conf
                    )

                self.compact_memory.append(thread_id, "user", message)
                prompt_load = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_prompt_tokens[thread_id] = (
                    self.thread_prompt_tokens.get(thread_id, 0) + prompt_load
                )

                profile_md = self.profile_store.read_text(user_id)
                ctx = self.compact_memory.context(thread_id)
                summary_text = str(ctx.get("summary", ""))
                system_context = (
                    f"Hồ sơ người dùng (User.md):\n{profile_md}\n\n"
                    f"Tóm tắt hội thoại trước:\n{summary_text}"
                )
                result = self.langchain_agent.invoke(
                    {
                        "messages": [
                            {"role": "system", "content": system_context},
                            {"role": "user", "content": message},
                        ]
                    },
                    config={"configurable": {"thread_id": thread_id}},
                )
                last_msg = result["messages"][-1]
                response_text = str(getattr(last_msg, "content", last_msg))
                self.compact_memory.append(thread_id, "assistant", response_text)
                turn_tokens = estimate_tokens(message) + estimate_tokens(response_text)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + turn_tokens
                return {
                    "response": response_text,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "token_usage": self.thread_tokens[thread_id],
                    "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
                    "compactions": self.compaction_count(thread_id),
                    "memory_size": self.memory_file_size(user_id),
                }
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline path with `User.md` persistence and compact memory."""

        min_conf = getattr(self.config, "confidence_threshold", 0.75)
        updates_with_conf = extract_profile_updates_with_confidence(
            message, min_confidence=min_conf
        )
        if updates_with_conf:
            updates = {k: v for k, (v, _c) in updates_with_conf.items()}
            confs = {k: c for k, (_v, c) in updates_with_conf.items()}
            self.profile_store.upsert_facts(
                user_id,
                updates=updates,
                confidences=confs,
                min_confidence=min_conf,
            )

        self.compact_memory.append(thread_id, "user", message)

        prompt_load = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_load
        )

        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        turn_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + turn_tokens

        return {
            "response": response,
            "user_id": user_id,
            "thread_id": thread_id,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            "compactions": self.compaction_count(thread_id),
            "memory_size": self.memory_file_size(user_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate context tokens carried into one turn: `User.md` + compact summary + recent kept messages."""

        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text = str(ctx.get("summary", ""))
        recent_messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]

        return (
            estimate_tokens(profile_text)
            + estimate_tokens(summary_text)
            + sum(estimate_tokens(m.get("content", "")) for m in recent_messages)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted `User.md` profile and compact thread context."""

        facts = self.profile_store.facts(user_id)
        raw_profile = self.profile_store.read_text(user_id)
        low = (message or "").lower()

        is_recall_query = "?" in low or any(
            kw in low
            for kw in (
                "nhắc lại",
                "mình tên gì",
                "tên mình là gì",
                "ở đâu",
                "làm nghề gì",
                "nghề hiện tại",
                "yêu thích",
                "nuôi con gì",
                "tóm tắt",
                "style trả lời",
                "kiểu trả lời",
                "dũngct",
            )
        )

        if is_recall_query:
            if facts:
                bullets: list[str] = []
                if "name" in facts:
                    bullets.append(f"- Tên: {facts['name']}")
                if "location" in facts:
                    bullets.append(f"- Nơi ở hiện tại: {facts['location']}")
                if "profession" in facts:
                    bullets.append(f"- Nghề nghiệp hiện tại: {facts['profession']}")
                if "drink" in facts or "food" in facts:
                    favs = []
                    if "drink" in facts:
                        favs.append(f"đồ uống: {facts['drink']}")
                    if "food" in facts:
                        favs.append(f"món ăn: {facts['food']}")
                    bullets.append(f"- Sở thích ẩm thực ({', '.join(favs)})")
                if "pet" in facts:
                    bullets.append(f"- Thú cưng: {facts['pet']}")
                if "response_style" in facts:
                    bullets.append(f"- Style trả lời: {facts['response_style']}")
                if "interests" in facts:
                    bullets.append(f"- Mối quan tâm chính: {facts['interests']}")
                return "Thông tin từ bộ nhớ bền vững User.md:\n" + "\n".join(bullets)

            if raw_profile.strip():
                return f"Thông tin từ User.md:\n{raw_profile.strip()}"

            return "Hiện chưa có thông tin trong User.md cho câu hỏi này."

        # Concise acknowledgement on non-question turns to keep short-term history clean
        updated = extract_profile_updates(message)
        if updated:
            keys_str = ", ".join(f"{k}={v}" for k, v in updated.items())
            return f"Đã lưu vào User.md ({keys_str}) và cập nhật ngữ cảnh."
        return "Đã ghi nhận và đối chiếu với hồ sơ User.md cùng ngữ cảnh nén."

    def _maybe_build_langchain_agent(self):
        """Optionally wire a live LangGraph agent with `User.md` tools and checkpointer."""

        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent

            store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persistent User.md markdown profile for a user."""
                return store.read_text(user_id)

            @tool
            def update_user_profile(user_id: str, key: str, value: str) -> str:
                """Upsert a structured fact into the user's persistent User.md profile."""
                store.upsert_fact(user_id, key, value)
                return store.read_text(user_id)

            chat_model = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()
            return create_react_agent(
                model=chat_model,
                tools=[read_user_profile, update_user_profile],
                checkpointer=checkpointer,
            )
        except Exception:
            return None