"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
def get_speech_synthesizer(
    engine_type: str = "glados", **kwargs: dict[str, Any]
) -> SpeechSynthesizerProtocol:
    """
    Factory function to get an instance of an audio synthesizer based on the specified engine type.

    Parameters:
        engine_type (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - "remote": Remote TTS via OpenAI-compatible API
            - <str>: Kokoro voice synthesizer using the specified voice
        **kwargs: Additional keyword arguments to pass to the synthesizer constructor

    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer

    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if engine_type.lower() == "glados":
        from ..TTS import tts_glados

        return tts_glados.SpeechSynthesizer()
    elif engine_type.lower() == "remote":
        from .remote_tts import RemoteSynthesizer

        return RemoteSynthesizer(**kwargs)

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if engine_type not in available_voices:
        raise ValueError(f"Voice '{engine_type}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=engine_type)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
