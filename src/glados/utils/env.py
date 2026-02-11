"""Environment variable override and sample rate utilities for GLaDOS configuration."""

import os

from loguru import logger

# Known sample rates by model family
SAMPLE_RATES: dict[str, int] = {
    "piper": 22050,
    "glados": 22050,
    "kokoro": 24000,
}


def _get_env_override(env_key: str, config_value: str | None) -> str | None:
    """Return env var if set, otherwise config value."""
    env_val = os.environ.get(env_key)
    if env_val is None:
        return config_value
    logger.info(f"Env override: {env_key}={env_val}")
    return env_val


def _get_env_bool(env_key: str, config_value: bool) -> bool:
    """Return env var as bool if set, otherwise config value."""
    env_val = os.environ.get(env_key)
    if env_val is None:
        return config_value
    result = env_val.lower() in ("true", "1", "yes")
    logger.info(f"Env override: {env_key}={env_val} -> {result}")
    return result


def _get_sample_rate(tts_model: str | None, tts: object) -> int:
    """Detect sample rate from model name patterns or TTS instance.

    Checks model name against known patterns first, falls back to tts.sample_rate.
    """
    if tts_model:
        model_lower = tts_model.lower()
        for pattern, rate in SAMPLE_RATES.items():
            if pattern in model_lower:
                return rate
    return getattr(tts, "sample_rate", 22050)
