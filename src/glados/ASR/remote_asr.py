"""Remote ASR transcriber using OpenAI-compatible API (e.g., Speaches)."""

import io
from pathlib import Path
import wave

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import requests


class RemoteTranscriber:
    """Speech-to-text transcriber that sends audio to a remote OpenAI-compatible transcription endpoint."""

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        asr_url: str,
        asr_model: str,
        asr_language: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout: float = 30.0,
    ) -> None:
        self.asr_url = asr_url
        self.asr_model = asr_model
        self.asr_language = asr_language
        self.sample_rate = sample_rate
        self.timeout = timeout

        logger.info(f"RemoteTranscriber initialized: {asr_url}, model={asr_model}")

    def _audio_to_wav_bytes(self, audio: NDArray[np.float32]) -> bytes:
        """Convert float32 audio array (values in [-1, 1]) to WAV bytes."""
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)
        return buffer.read()

    def _build_form_data(self) -> dict[str, str]:
        data: dict[str, str] = {"model": self.asr_model}
        if self.asr_language:
            data["language"] = self.asr_language
        return data

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """Transcribe audio array by sending to remote endpoint."""
        logger.debug(f"Transcribing {len(audio)} samples ({len(audio) / self.sample_rate:.2f}s)")

        wav_bytes = self._audio_to_wav_bytes(audio)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}

        response = requests.post(
            self.asr_url,
            files=files,
            data=self._build_form_data(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("text", "").strip()
        logger.debug(f"Transcription result: '{text}'")
        return text

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file by sending to remote endpoint."""
        suffix = audio_path.suffix.lower()
        content_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/m4a",
        }
        content_type = content_types.get(suffix, "audio/wav")

        with open(audio_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": (audio_path.name, file_bytes, content_type)}

        response = requests.post(
            self.asr_url,
            files=files,
            data=self._build_form_data(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("text", "").strip()
