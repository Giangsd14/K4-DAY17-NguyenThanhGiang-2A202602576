from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A (Baseline): within-session memory only, no persistent `User.md`, no compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting for a turn."""

        if not self.force_offline and self.langchain_agent is not None:
            try:
                session = self.sessions.setdefault(thread_id, SessionState())
                session.messages.append({"role": "user", "content": message})
                turn_prompt_tokens = sum(
                    estimate_tokens(m.get("content", "")) for m in session.messages
                )
                session.prompt_tokens_processed += turn_prompt_tokens

                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                last_msg = result["messages"][-1]
                response_text = str(getattr(last_msg, "content", last_msg))
                session.messages.append({"role": "assistant", "content": response_text})
                session.token_usage += estimate_tokens(message) + estimate_tokens(response_text)
                return {
                    "response": response_text,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "token_usage": session.token_usage,
                    "prompt_tokens_processed": session.prompt_tokens_processed,
                    "compactions": 0,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def memory_file_size(self, user_id: str) -> int:
        return 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior restricted strictly to the current `thread_id`."""

        session = self.sessions.setdefault(thread_id, SessionState())
        prior_messages = list(session.messages)
        session.messages.append({"role": "user", "content": message})

        # Baseline re-processes the entire raw thread history on every single turn
        turn_prompt_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in session.messages
        )
        session.prompt_tokens_processed += turn_prompt_tokens

        in_thread_facts: dict[str, str] = {}
        for msg in prior_messages:
            if msg.get("role") == "user":
                in_thread_facts.update(extract_profile_updates(msg.get("content", "")))

        low = (message or "").lower()
        is_recall_query = "?" in low or any(
            kw in low
            for kw in (
                "nhắc lại",
                "mình tên gì",
                "tên mình là gì",
                "ở đâu",
                "làm nghề gì",
                "yêu thích",
                "nuôi con gì",
                "tóm tắt",
            )
        )

        if is_recall_query:
            if in_thread_facts:
                parts = [f"{k}: {v}" for k, v in in_thread_facts.items()]
                response = "Trong phiên hiện tại mình ghi nhận: " + ", ".join(parts) + "."
            else:
                response = (
                    "Mình chỉ có bộ nhớ trong cùng phiên (within-session) nên không nhớ "
                    "thông tin từ các phiên trước."
                )
        else:
            response = "Đã ghi nhận thông tin trong phiên hội thoại hiện tại."

        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)

        return {
            "response": response,
            "thread_id": thread_id,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire a live LangGraph agent with `InMemorySaver` when credentials exist."""

        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent

            chat_model = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()
            return create_react_agent(model=chat_model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None