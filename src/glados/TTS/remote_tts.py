"""Remote TTS synthesizer using OpenAI-compatible API (e.g., Speaches)."""

import io
import wave

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import requests


class RemoteSynthesizer:
    """Text-to-speech synthesizer that sends text to a remote OpenAI-compatible speech endpoint."""

    DEFAULT_SAMPLE_RATE = 24000

    def __init__(
        self,
        tts_url: str,
        tts_model: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout: float = 30.0,
    ) -> None:
        self._tts_url = tts_url
        self._tts_model = tts_model
        self._sample_rate = sample_rate
        self._timeout = timeout

        logger.info(f"RemoteSynthesizer initialized: {tts_url}, model={tts_model}")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Generate speech audio by sending text to the remote TTS endpoint.

        Parameters:
            text: The text to synthesize into speech.

        Returns:
            NDArray[np.float32]: Audio samples in float32 format, values in [-1, 1].
        """
        logger.debug(f"Synthesizing {len(text)} chars via remote TTS")

        response = requests.post(
            self._tts_url,
            json={
                "model": self._tts_model,
                "input": text,
                "voice": "default",
                "response_format": "wav",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        return self._wav_bytes_to_audio(response.content)

    def _wav_bytes_to_audio(self, wav_bytes: bytes) -> NDArray[np.float32]:
        """Convert WAV bytes to float32 audio array (values in [-1, 1])."""
        buffer = io.BytesIO(wav_bytes)
        with wave.open(buffer, "rb") as wav_file:
            self._sample_rate = wav_file.getframerate()
            raw_frames = wav_file.readframes(wav_file.getnframes())
            audio_int16 = np.frombuffer(raw_frames, dtype=np.int16)

        return audio_int16.astype(np.float32) / 32768.0
