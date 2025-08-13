import os
import threading
import time
import uuid
import queue
import tempfile
from typing import Callable, Optional, List

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None


class InteractionOrchestrator:
    """Thread-safe coordinator for conversational interrupts and playback control.

    Responsibilities:
    - Track the active response cycle (response_id)
    - Coordinate generation/playback abort requests (stop, barge-in)
    - Record which sentences were already spoken to avoid repetition on resume
    - Hold optional barge-in text for continuation prompts
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_response_id: Optional[str] = None
        self._abort_generation: bool = False
        self._abort_playback: bool = False
        self._barge_in_text: Optional[str] = None
        self._spoken_sentences: List[str] = []
        self._current_speech_queue: Optional["SpeechQueue"] = None

    # Lifecycle ---------------------------------------------------------------
    def begin_response(self, response_id: Optional[str] = None) -> str:
        with self._lock:
            rid = response_id or str(uuid.uuid4())
            self._active_response_id = rid
            self._abort_generation = False
            self._abort_playback = False
            self._barge_in_text = None
            self._spoken_sentences.clear()
            return rid

    def end_response(self, response_id: Optional[str] = None) -> None:
        with self._lock:
            if response_id is None or response_id == self._active_response_id:
                self._active_response_id = None
                self._abort_generation = False
                self._abort_playback = False
                self._barge_in_text = None
                self._spoken_sentences.clear()
                self._current_speech_queue = None

    # State queries -----------------------------------------------------------
    def has_active_response(self) -> bool:
        with self._lock:
            return self._active_response_id is not None

    def get_last_spoken(self, n: int = 1) -> List[str]:
        with self._lock:
            if n <= 0:
                return []
            return self._spoken_sentences[-n:].copy()

    # Recording ---------------------------------------------------------------
    def record_spoken_sentence(self, sentence: str) -> None:
        if not sentence:
            return
        with self._lock:
            self._spoken_sentences.append(sentence.strip())
            # Keep only the last few for compact prompts
            if len(self._spoken_sentences) > 10:
                self._spoken_sentences = self._spoken_sentences[-10:]

    def bind_speech_queue(self, speech_queue: Optional["SpeechQueue"]) -> None:
        with self._lock:
            self._current_speech_queue = speech_queue

    # Interrupts --------------------------------------------------------------
    def request_stop(self) -> None:
        """Hard stop: cancel playback and generation; remain idle."""
        with self._lock:
            self._abort_playback = True
            self._abort_generation = True
        self._cancel_queue_if_any()

    def request_barge_in(self, user_text: str) -> None:
        """Stop immediately and mark user_text for continuation prompt."""
        with self._lock:
            self._abort_playback = True
            self._abort_generation = True
            self._barge_in_text = (user_text or "").strip()
        self._cancel_queue_if_any()

    def consume_barge_in_text(self) -> Optional[str]:
        with self._lock:
            text = self._barge_in_text
            self._barge_in_text = None
            return text

    def should_abort_generation(self) -> bool:
        with self._lock:
            return self._abort_generation

    def should_abort_playback(self) -> bool:
        with self._lock:
            return self._abort_playback

    def clear_abort_flags(self) -> None:
        with self._lock:
            self._abort_generation = False
            self._abort_playback = False

    # Internal ----------------------------------------------------------------
    def _cancel_queue_if_any(self) -> None:
        sq = None
        with self._lock:
            sq = self._current_speech_queue
        if sq is not None:
            try:
                sq.cancel_all()
            except Exception:
                pass


class AudioPlaybackController:
    """Cancellable local WAV playback using sounddevice when available.

    Fallback to no-op if sounddevice/soundfile are unavailable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def play_wav_blocking(self, wav_path: str, external_stop_event: Optional[threading.Event] = None) -> None:
        if not os.path.exists(wav_path):
            return
        if sd is None or sf is None:
            # As a safe fallback, do a simple blocking sleep proportional to file length guess
            time.sleep(0.1)
            return

        try:
            with sf.SoundFile(wav_path, 'r') as f:
                samplerate = f.samplerate
                channels = f.channels
                blocksize = 1024

                def callback(outdata, frames, time_info, status):
                    # This callback is required but we drive the stream manually; leaving for completeness
                    return

                stop_event = external_stop_event or threading.Event()
                with sd.OutputStream(samplerate=samplerate, channels=channels, dtype='float32') as stream:
                    while True:
                        if stop_event.is_set():
                            break
                        data = f.read(blocksize, dtype='float32')
                        if data is None or len(data) == 0:
                            break
                        stream.write(data)
        except Exception:
            # Swallow playback errors silently to avoid breaking primary flow
            pass

    def stop(self) -> None:
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
            th = self._thread
            self._thread = None
            self._stop_event = None
        if th is not None:
            try:
                th.join(timeout=0.5)
            except Exception:
                pass


class SpeechQueue:
    """Sentence-level TTS queue with cancellation and small prebuffer.

    Single responsibility: accept text segments, synthesize to WAV, and play/send
    them sequentially until cancelled.
    """

    def __init__(
        self,
        playback_controller: AudioPlaybackController,
        tts_engine_provider: Callable[[], Optional[object]],
        send_audio_to_pi_fn: Callable[[str, str], None],
        send_stop_to_pi_fn: Optional[Callable[[], None]] = None,
        prebuffer_sentences: int = 1,
        language: str = "en",
        speaker_wav_path: Optional[str] = None,
        default_speaker: str = "Ana Florence",
    ) -> None:
        self._playback = playback_controller
        self._tts_engine_provider = tts_engine_provider
        self._send_audio_to_pi = send_audio_to_pi_fn
        self._send_stop_to_pi = send_stop_to_pi_fn
        self._prebuffer = max(0, int(prebuffer_sentences))
        self._language = language
        self._speaker_wav_path = speaker_wav_path
        self._default_speaker = default_speaker

        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._target_device = "local"
        self._mute = False
        # Optional callbacks
        self.on_playback_start: Optional[Callable[[], None]] = None
        self.on_playback_end: Optional[Callable[[], None]] = None
        self.on_sentence_spoken: Optional[Callable[[str], None]] = None

    def configure(self, target_device: str, mute: bool) -> None:
        self._target_device = target_device
        self._mute = bool(mute)

    def enqueue(self, sentence_text: str) -> None:
        if not sentence_text or not sentence_text.strip():
            return
        self._queue.put((str(uuid.uuid4()), sentence_text.strip()))
        if not self._started and self._queue.qsize() >= self._prebuffer:
            self._start_worker()

    def cancel_all(self) -> None:
        self._stop_event.set()
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass
        # Stop active playback
        try:
            self._playback.stop()
        except Exception:
            pass
        # Inform Pi clients to stop playback
        try:
            if self._send_stop_to_pi is not None:
                self._send_stop_to_pi()
        except Exception:
            pass

    def _start_worker(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _run_worker(self) -> None:
        try:
            started_callback_fired = False
            while not self._stop_event.is_set():
                try:
                    item_id, sentence = self._queue.get(timeout=0.1)
                except queue.Empty:
                    # Nothing to say; if queue drained and previously started, keep worker alive a little
                    if self._queue.empty():
                        time.sleep(0.05)
                    continue

                wav_path = self._synthesize_sentence(sentence)
                if wav_path:
                    try:
                        if not started_callback_fired and self.on_playback_start is not None:
                            try:
                                self.on_playback_start()
                            except Exception:
                                pass
                            started_callback_fired = True
                        if self._target_device in ("pi", "all"):
                            # Send audio to Pi; Pi side handles its own playback + STT gating
                            self._send_audio_to_pi(wav_path, sentence)
                        if not self._mute and self._target_device in ("local", "all"):
                            # Local cancellable playback
                            self._playback.play_wav_blocking(wav_path, self._stop_event)
                    finally:
                        try:
                            if os.path.exists(wav_path):
                                os.remove(wav_path)
                        except Exception:
                            pass

                    # Notify sentence finished playing
                    if self.on_sentence_spoken is not None:
                        try:
                            self.on_sentence_spoken(sentence)
                        except Exception:
                            pass

                # Early exit if cancelled mid-queue
                if self._stop_event.is_set():
                    break
        finally:
            self._started = False
            if self.on_playback_end is not None:
                try:
                    self.on_playback_end()
                except Exception:
                    pass

    def _synthesize_sentence(self, text: str) -> Optional[str]:
        """Generate a temporary WAV file for the given text using the provided TTS engine.

        Returns the WAV path or None on failure.
        """
        try:
            tts_engine = self._tts_engine_provider()
            if tts_engine is None:
                return None

            # Ensure temp file path
            fd, wav_path = tempfile.mkstemp(prefix="fred_tts_", suffix=".wav")
            os.close(fd)

            # Prefer cloning from speaker WAV if available
            if self._speaker_wav_path and os.path.exists(self._speaker_wav_path):
                tts_engine.tts_to_file(
                    text=text,
                    speaker_wav=self._speaker_wav_path,
                    language=self._language,
                    file_path=wav_path,
                )
            else:
                tts_engine.tts_to_file(
                    text=text,
                    speaker=self._default_speaker,
                    language=self._language,
                    file_path=wav_path,
                )
            return wav_path
        except Exception:
            return None


