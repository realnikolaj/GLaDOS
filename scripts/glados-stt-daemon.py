#!/usr/bin/env python3
"""
GLaDOS STT Daemon

Speech-to-text daemon using GLaDOS's VAD for voice activity detection
and Speaches for transcription. Exposes a Unix socket interface for
Hammerspoon or other clients.

Key design (learned from GLaDOS speech_listener.py):
- Audio callback ALWAYS queues audio, regardless of state
- VAD runs on every chunk
- Transcription happens AFTER silence is detected, not during recording
- No audio is ever lost

Usage:
    # Activate GLaDOS conda env first
    micromamba activate glados
    
    # Run daemon
    python glados-stt-daemon.py

    # Control via socket
    echo "start" | nc -U /tmp/glados-stt.sock
    echo "stop" | nc -U /tmp/glados-stt.sock

Socket Protocol:
    start       → "started" (begin recording, auto-transcribe on silence)
    stop        → "<transcript>" (stop session, return all accumulated text)
    cancel      → "cancelled" (discard everything)
    status      → "idle|listening|recording|processing"
    
    Push messages (daemon → client):
    transcript:<text>   → Sent when silence detected and transcription complete
"""

import asyncio
import io
import os
import sys
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import requests

# Import VAD from GLaDOS
try:
    from glados.audio_io.vad import VAD
except ImportError:
    print("ERROR: Cannot import from glados. Make sure you're in the GLaDOS conda env:")
    print("  micromamba activate glados")
    print("  cd ~/git/GLaDOS && pip install -e .")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Speaches endpoint (same as GLaDOS remote ASR config)
SPEACHES_HOST = os.environ.get("SPEACHES_HOST", "x3d-v.tail1a9c7.ts.net:34331")
SPEACHES_URL = f"http://{SPEACHES_HOST}/v1/audio/transcriptions"
STT_MODEL = os.environ.get("STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")

# Audio settings (match GLaDOS)
SAMPLE_RATE = 16000
VAD_SIZE_MS = 32  # Milliseconds per VAD chunk
VAD_SAMPLES = int(SAMPLE_RATE * VAD_SIZE_MS / 1000)  # 512 samples

# Timing settings (from GLaDOS speech_listener.py)
BUFFER_SIZE_MS = 800   # Pre-speech buffer
PAUSE_LIMIT_MS = 640   # Silence before processing
VAD_THRESHOLD = 0.5    # Speech probability threshold

BUFFER_CHUNKS = BUFFER_SIZE_MS // VAD_SIZE_MS  # 25 chunks
PAUSE_CHUNKS = PAUSE_LIMIT_MS // VAD_SIZE_MS   # 20 chunks

# Socket
SOCKET_PATH = "/tmp/glados-stt.sock"
LOG_PATH = "/tmp/glados-stt.log"


# =============================================================================
# Logging
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except:
        pass


# =============================================================================
# Transcription
# =============================================================================

def transcribe_audio(audio: np.ndarray) -> str:
    """Send audio to Speaches and return transcript."""
    if len(audio) == 0:
        return ""
    
    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val < 1e-10:
        log("⚠️ Silent audio, skipping transcription")
        return ""
    audio = audio / max_val
    
    # Convert to WAV bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    wav_bytes = buffer.read()
    
    # Send to Speaches
    duration = len(audio) / SAMPLE_RATE
    log(f"📤 Sending {duration:.1f}s audio to Speaches...")
    
    try:
        response = requests.post(
            SPEACHES_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": STT_MODEL, "language": STT_LANGUAGE},
            timeout=30
        )
        response.raise_for_status()
        text = response.json().get("text", "").strip()
        log(f"📝 Transcript: '{text}'")
        return text
    except Exception as e:
        log(f"❌ Transcription error: {e}")
        return ""


# =============================================================================
# State Machine
# =============================================================================

class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"  # Waiting for speech
    RECORDING = "recording"  # Speech detected, accumulating
    PROCESSING = "processing"  # Transcribing


@dataclass
class Session:
    """Active recording session."""
    state: State = State.IDLE
    
    # Pre-speech circular buffer (GLaDOS pattern)
    pre_buffer: deque = field(default_factory=lambda: deque(maxlen=BUFFER_CHUNKS))
    
    # Accumulated speech samples
    samples: list = field(default_factory=list)
    
    # Silence gap counter
    gap_counter: int = 0
    
    # All transcripts this session
    transcripts: list = field(default_factory=list)
    
    # Connected client for pushing transcripts
    client_writer: Optional[asyncio.StreamWriter] = None
    
    def reset(self):
        """Reset for new session."""
        self.pre_buffer.clear()
        self.samples.clear()
        self.gap_counter = 0
        self.transcripts.clear()
        self.state = State.IDLE


# =============================================================================
# Audio Processor
# =============================================================================

class AudioProcessor:
    """
    Handles audio capture and VAD processing.
    
    Key insight from GLaDOS: the audio callback ALWAYS queues audio.
    State changes don't affect audio capture - only how we process it.
    """
    
    def __init__(self):
        self.vad = VAD()
        self.session = Session()
        self.audio_queue: asyncio.Queue = None
        self.stream: sd.InputStream = None
        self.loop: asyncio.AbstractEventLoop = None
        
    def start_session(self, client_writer: asyncio.StreamWriter) -> str:
        """Start a new recording session."""
        if self.session.state != State.IDLE:
            return "already_recording"
        
        self.session.reset()
        self.session.client_writer = client_writer
        self.session.state = State.LISTENING
        
        log("▶️ Session started (listening for speech)")
        return "started"
    
    def stop_session(self) -> str:
        """Stop session and return all transcripts."""
        if self.session.state == State.IDLE:
            return ""
        
        # If we have uncommitted audio, transcribe it
        if self.session.samples:
            log("🔄 Processing remaining audio...")
            audio = np.concatenate(self.session.samples)
            transcript = transcribe_audio(audio)
            if transcript:
                self.session.transcripts.append(transcript)
        
        # Combine all transcripts
        full_text = " ".join(self.session.transcripts)
        
        self.session.reset()
        log(f"⏹️ Session stopped: '{full_text[:50]}...' ({len(full_text)} chars)")
        
        return full_text if full_text else "(empty)"
    
    def cancel_session(self) -> str:
        """Cancel session, discard everything."""
        self.session.reset()
        log("🚫 Session cancelled")
        return "cancelled"
    
    def get_status(self) -> str:
        """Get current state."""
        return self.session.state.value
    
    async def push_transcript(self, text: str):
        """Push transcript to connected client."""
        if self.session.client_writer:
            try:
                msg = f"transcript:{text}\n"
                self.session.client_writer.write(msg.encode())
                await self.session.client_writer.drain()
                log(f"📨 Pushed: '{text[:30]}...'")
            except Exception as e:
                log(f"⚠️ Failed to push: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback - ALWAYS queues audio.
        This is the key fix: no state check here.
        """
        if status:
            log(f"⚠️ Audio status: {status}")
        
        # Always queue, let the processor decide what to do
        if self.loop and self.audio_queue:
            data = indata[:, 0].copy()  # Mono
            self.loop.call_soon_threadsafe(
                self.audio_queue.put_nowait,
                data
            )
    
    async def _process_audio_chunk(self, chunk: np.ndarray):
        """Process a single audio chunk with VAD."""
        # Run VAD
        vad_prob = self.vad(np.expand_dims(chunk, 0))
        is_speech = vad_prob > VAD_THRESHOLD
        
        state = self.session.state
        
        if state == State.IDLE:
            # Not in a session, ignore
            return
        
        elif state == State.LISTENING:
            # Waiting for speech - maintain pre-buffer
            self.session.pre_buffer.append(chunk)
            
            if is_speech:
                # Speech detected! Start recording
                log("🎤 Speech detected")
                self.session.samples = list(self.session.pre_buffer)
                self.session.state = State.RECORDING
                self.session.gap_counter = 0
        
        elif state == State.RECORDING:
            # Accumulating speech
            self.session.samples.append(chunk)
            
            if not is_speech:
                self.session.gap_counter += 1
                
                if self.session.gap_counter >= PAUSE_CHUNKS:
                    # Silence detected - transcribe
                    log(f"🔇 Silence detected ({PAUSE_LIMIT_MS}ms)")
                    self.session.state = State.PROCESSING
                    
                    # Transcribe accumulated audio
                    audio = np.concatenate(self.session.samples)
                    transcript = transcribe_audio(audio)
                    
                    if transcript:
                        self.session.transcripts.append(transcript)
                        # Push to client
                        await self.push_transcript(transcript)
                    
                    # Reset for next utterance (stay in session)
                    self.session.samples.clear()
                    self.session.pre_buffer.clear()
                    self.session.gap_counter = 0
                    self.session.state = State.LISTENING
                    log("👂 Listening for next utterance...")
            else:
                self.session.gap_counter = 0
        
        elif state == State.PROCESSING:
            # Still capture audio while processing
            # It goes to pre_buffer for next utterance
            self.session.pre_buffer.append(chunk)
    
    async def run(self):
        """Main audio processing loop."""
        self.loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=VAD_SAMPLES,
            callback=self._audio_callback
        )
        self.stream.start()
        log(f"🎧 Audio stream started (SR={SAMPLE_RATE}, chunk={VAD_SAMPLES})")
        
        try:
            while True:
                chunk = await self.audio_queue.get()
                await self._process_audio_chunk(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            self.stream.stop()
            self.stream.close()
            log("🔇 Audio stream stopped")


# =============================================================================
# Socket Server
# =============================================================================

class Server:
    def __init__(self):
        self.processor = AudioProcessor()
        self.process_task: asyncio.Task = None
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection."""
        try:
            data = await reader.read(1024)
            command = data.decode().strip().lower()
            log(f"📨 Command: {command}")
            
            if command == "start":
                response = self.processor.start_session(writer)
                writer.write(f"{response}\n".encode())
                await writer.drain()
                
                if response == "started":
                    # Keep connection open for transcript pushes
                    # Wait until session ends or client disconnects
                    try:
                        while self.processor.session.state != State.IDLE:
                            await asyncio.sleep(0.1)
                    except:
                        pass
                    return  # Don't close writer, let stop/cancel do it
            
            elif command == "stop":
                transcript = self.processor.stop_session()
                writer.write(f"{transcript}\n".encode())
                await writer.drain()
            
            elif command == "cancel":
                response = self.processor.cancel_session()
                writer.write(f"{response}\n".encode())
                await writer.drain()
            
            elif command == "status":
                response = self.processor.get_status()
                writer.write(f"{response}\n".encode())
                await writer.drain()
            
            elif command == "quit":
                writer.write(b"goodbye\n")
                await writer.drain()
                asyncio.get_event_loop().stop()
            
            else:
                writer.write(f"unknown: {command}\n".encode())
                await writer.drain()
        
        except Exception as e:
            log(f"❌ Client error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
    
    async def run(self):
        """Start the server."""
        # Clean up old socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        
        # Start audio processor
        self.process_task = asyncio.create_task(self.processor.run())
        
        # Start socket server
        server = await asyncio.start_unix_server(
            self.handle_client,
            path=SOCKET_PATH
        )
        os.chmod(SOCKET_PATH, 0o666)
        
        log("=" * 60)
        log("GLaDOS STT Daemon")
        log(f"  Speaches: {SPEACHES_URL}")
        log(f"  Model: {STT_MODEL}")
        log(f"  Socket: {SOCKET_PATH}")
        log(f"  VAD threshold: {VAD_THRESHOLD}")
        log(f"  Pre-buffer: {BUFFER_SIZE_MS}ms")
        log(f"  Silence timeout: {PAUSE_LIMIT_MS}ms")
        log("=" * 60)
        log("Commands: start | stop | cancel | status | quit")
        log("Ready.")
        
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            self.process_task.cancel()
            server.close()
            await server.wait_closed()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
            log("Daemon stopped.")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GLaDOS STT Daemon")
    parser.add_argument("--test", action="store_true", help="Test VAD and exit")
    args = parser.parse_args()
    
    if args.test:
        log("Testing VAD...")
        vad = VAD()
        test_audio = np.random.randn(VAD_SAMPLES).astype(np.float32) * 0.01
        result = vad(np.expand_dims(test_audio, 0))
        log(f"VAD test result: {result:.3f} (should be low for noise)")
        log("✓ VAD working")
        return
    
    server = Server()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        log("Interrupted.")


if __name__ == "__main__":
    main()
