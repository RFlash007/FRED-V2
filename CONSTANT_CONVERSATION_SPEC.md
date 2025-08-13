# F.R.E.D. Constant Conversation (A+) — Behavioral Spec

Purpose
- Enable natural, "Jarvis-like" continuous conversation with:
  - Speak-as-generated (sentence queue TTS)
  - Instant barge-in (voice or UI) with seamless continuation
  - Hard stop on command
- Keep components decoupled and single-purpose for clarity and future reuse.

Core Concepts
- States: Idle → Generating → Speaking
  - Speaking is cancelable at any time (preempted by user input)
  - Generating can be aborted to incorporate a user’s interjection
- Output segmentation: Responses stream from the LLM and are split into full sentences; each sentence is synthesized and played/sent as soon as available.

Key Components (single responsibility)
- InteractionOrchestrator (conversation_orchestrator.py)
  - Tracks the active response cycle (response_id)
  - Flags to abort generation/playback
  - Captures barge-in text
  - Stores last-spoken sentences (to avoid repetition on continuation)
- SpeechQueue (conversation_orchestrator.py)
  - FIFO of sentence items
  - Synthesizes to WAV and plays locally; also sends to Pi
  - Cancellable mid-queue; supports prebuffer before speaking (set to 1 sentence)
  - Callbacks: on_playback_start, on_sentence_spoken, on_playback_end
- AudioPlaybackController (conversation_orchestrator.py)
  - Cancellable local playback using sounddevice/soundfile; safe fallbacks

Primary Flow (/chat in app.py)
1) Receive user input; build LLM messages with system prompt + history + optional agenda/context.
2) Create response_id via orchestrator; set up SpeechQueue (prebuffer=1 sentence):
   - on_playback_start → STT speaking state TRUE
   - on_playback_end → STT speaking state FALSE
   - on_sentence_spoken → record sentence in orchestrator
3) Stream LLM output (ollama_manager.chat_concurrent_safe) and:
   - Strip <think> blocks
   - Append clean text to response stream (SSE to UI)
   - Segment into sentences and enqueue to SpeechQueue
   - On orchestrator.should_abort_generation() → break
4) On abort with barge-in text:
   - Build continuation prompt:
     - “You were interrupted mid-response… User said: '{text}'. Avoid repeating already spoken sentence: '{last_spoken}'. Continue seamlessly.”
   - Append assistant-so-far + continuation prompt; resume streaming as in step 3
5) Finalize: enqueue any trailing partial sentence; end response lifecycle.

Sentence Segmentation
- Simple rule-based: split on [. ! ?] with whitespace; keep last partial as tail until completed.
- Queue prebuffer set to 1 sentence for minimal latency.

Barge-in Semantics
- Voice while speaking (stt_service.py):
  - “stop” → immediate stop of audio + generation; no auto-continue
  - “fred …” → immediate stop; capture remainder as user context; recompute and continue
  - During speaking, STT runs in “barge-in mode”: limited grammar, stricter VAD; no full transcription chatter
- Voice while generating (not speaking):
  - Same logic; generation aborts and continuation resumes with context
- UI buttons (templates/index.html → static/script.js → /api/interrupt):
  - Stop → same as “stop”
  - Stop + Continue (with text) → same as “fred …”

Prompt Integration (continuations)
- Append:
  - Assistant content generated so far (as context)
  - Meta block: interruption notice, user interjection text, and last-spoken sentence hint to reduce repetition
- Continue generation; sentence queue resumes speaking seamlessly

Pi/WebRTC Integration
- Outbound audio to Pi: `socketio.emit('fred_audio', { audio_data, text, format: 'wav' })`
- Stop signal to Pi: `socketio.emit('fred_audio_stop', { message: 'stop' })`
- WebRTC server relays both to connected Pi data channels:
  - Audio → `[AUDIO_BASE64:wav]{base64}`
  - Stop → `[AUDIO_STOP]`
- Pi client (`pi_client/fred_pi_client.py`):
  - Plays via `Popen` (aplay/paplay/mpg123); stores process handle
  - On `[AUDIO_STOP]`, terminates player promptly; cleans temp files

APIs & Events
- HTTP
  - POST `/chat`: streams JSON lines with { response } chunks to UI
  - POST `/api/interrupt`: { action: 'stop' | 'continue', text?: string }
- SocketIO to UI/WebRTC server
  - `fred_audio` (send audio to Pi)
  - `fred_audio_stop` (stop Pi audio)
- STT callback to backend
  - Voice “stop” or “fred …” forwarded through `process_transcription` to orchestrator

Constraints & Guarantees
- Uses centralized Ollama connection manager for all model calls to avoid multiple clients and memory thrash.
- Playback and generation are always cancelable; cancellation latency targets <200ms in practice.
- No auditory cues or fluff; silent continuation—LLM integrates context naturally from the prompt.

Edge Cases
- Multiple interrupts quickly: last interrupt wins; old response_id cancelled (queue flushed; generation aborts)
- Partial sentence at stream end: enqueued once completed or on finalization
- If the barge-in has no content after “fred”: no-op (or silent acknowledgment), waiting for content

Success Criteria
- “Stop” halts audio and generation immediately; remains idle
- “Fred …” halts and resumes within ~1–2s, continuing seamlessly with user’s additions
- Minimal or no repeated sentences across continuation boundaries

Test Checklist (manual)
- Interrupt early/middle/late during speaking; verify cancellation and state recovery
- “Stop” vs “Stop + Continue” via UI; verify parity with voice
- Windows + Pi playback termination and restart
- Rapid repeated interrupts do not deadlock or duplicate speech

Extensibility
- Pluggable segmentation and TTS backends via `SpeechQueue`
- Swap barge-in detection (e.g., add dedicated KWS) without touching orchestrator or queue
- Add AEC or hardware push-to-interrupt later without changing core semantics

File Map (where behaviors live)
- `conversation_orchestrator.py`: Orchestrator, SpeechQueue, AudioPlaybackController
- `app.py`: streaming pipeline, sentence segmentation, interrupt/resume logic, `/api/interrupt`
- `stt_service.py`: barge-in mode during speaking (detect “stop” / “fred …”)
- `webrtc_server.py`: forwards audio/stop events between backend and Pi clients
- `pi_client/fred_pi_client.py`: playback with `Popen`, immediate termination on stop
- `templates/index.html`, `static/script.js`, `static/style.css`: Stop / Stop+Continue UI

Notes
- Prebuffer is set to 1 sentence to maximize perceived responsiveness.
- Unspoken streamed text is discarded on interrupt to avoid drift; last-spoken sentence is used to reduce repetition.
- Orchestrator is the single source of truth for aborts and barge-in data.
