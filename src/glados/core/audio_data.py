"""Core audio data structures for GLaDOS voice assistant.

This module defines message classes used for audio processing and communication
between different components of the voice assistant pipeline.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    """Audio message container for TTS output.

    Args:
        audio: Generated audio samples as float32 array
        text: Associated text that was synthesized
        is_eos: Flag indicating end of speech stream
        audio_stream: Optional streaming generator for chunked playback
    """

    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
    audio_stream: Iterator[NDArray[np.float32]] | None = field(default=None, repr=False)


@dataclass
class AudioInputMessage:
    """Audio input message container for ASR processing.

    Args:
        audio_sample: Raw audio input samples as float32 array
        vad_confidence: Voice activity detection confidence flag
    """

    audio_sample: NDArray[np.float32]
    vad_confidence: bool = False
