from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    upstage_api_key: str
    jev_model: str = "typesafe/jev-1.13"
    jev_url: str = "https://openrouter.ai/api/alpha/decisions"
    solar_model: str = "solar-pro3"
    solar_url: str = "https://api.upstage.ai/v1/chat/completions"
    request_timeout: float = 90.0
    max_history_turns: int = 8
    fast_guided_steps: int = 1
    full_guided_steps: int = 3
    adaptive_max_guided_steps: int = 6
    default_run_mode: str = "fast"
    debug: bool = False
    jev_batch_size: int = 8
    conversation_log_enabled: bool = True
    conversation_log_dir: str = "logs/conversations"

    @classmethod
    def load(cls, env_path: str | Path | None = None) -> "Config":
        if env_path is None:
            env_path = Path.cwd() / ".env"
        load_dotenv(env_path, override=False)

        def as_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def as_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        run_mode = os.getenv("DEFAULT_RUN_MODE", "fast").strip().lower()
        if run_mode not in {"fast", "full"}:
            run_mode = "fast"

        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            upstage_api_key=os.getenv("UPSTAGE_API_KEY", "").strip(),
            jev_model=os.getenv("JEV_MODEL", "typesafe/jev-1.13").strip(),
            jev_url=os.getenv("JEV_URL", "https://openrouter.ai/api/alpha/decisions").strip(),
            solar_model=os.getenv("SOLAR_MODEL", "solar-pro3").strip(),
            solar_url=os.getenv("SOLAR_URL", "https://api.upstage.ai/v1/chat/completions").strip(),
            request_timeout=as_float("REQUEST_TIMEOUT", 90.0),
            max_history_turns=max(1, as_int("MAX_HISTORY_TURNS", 8)),
            fast_guided_steps=max(1, as_int("FAST_GUIDED_STEPS", 1)),
            full_guided_steps=max(1, as_int("FULL_GUIDED_STEPS", 3)),
            adaptive_max_guided_steps=max(2, as_int("ADAPTIVE_MAX_GUIDED_STEPS", 6)),
            default_run_mode=run_mode,
            debug=os.getenv("DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"},
            jev_batch_size=max(1, min(20, as_int("JEV_BATCH_SIZE", 8))),
            conversation_log_enabled=os.getenv("CONVERSATION_LOG_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"},
            conversation_log_dir=os.getenv("CONVERSATION_LOG_DIR", "logs/conversations").strip() or "logs/conversations",
        )

    def validate(self) -> list[str]:
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.upstage_api_key:
            missing.append("UPSTAGE_API_KEY")
        return missing
