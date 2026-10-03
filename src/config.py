from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    confidence_threshold: float = 0.75


def _load_dotenv_file(dotenv_path: Path) -> None:
    """Load key-value pairs from `.env` using python-dotenv or a lightweight fallback."""

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)
        return
    except ImportError:
        pass

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _resolve_provider_credentials(provider: str) -> tuple[str | None, str | None]:
    """Resolve (api_key, base_url) from environment variables for a normalized provider."""

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")
    if provider == "custom":
        return (
            os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            os.getenv("CUSTOM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        )
    if provider == "gemini":
        return (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            None,
        )
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY"), os.getenv("ANTHROPIC_BASE_URL")
    if provider == "ollama":
        return None, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return (
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    return None, None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv_file(root / ".env")

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "220"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))
    confidence_threshold = float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.75"))

    raw_provider = os.getenv("LLM_PROVIDER", "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    api_key, base_url = _resolve_provider_credentials(provider)

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    judge_raw_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_provider = normalize_provider(judge_raw_provider)
    judge_model_name = os.getenv("JUDGE_MODEL") or (
        model_name if judge_provider == provider else DEFAULT_MODELS[judge_provider]
    )
    judge_temp = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge_api_key, judge_base_url = _resolve_provider_credentials(judge_provider)

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temp,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_cfg,
        judge_model=judge_cfg,
        confidence_threshold=confidence_threshold,
    )