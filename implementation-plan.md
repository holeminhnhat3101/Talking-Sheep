# Talking Sheep — Remaining Implementation Plan

## Scope

Tasks 1 and 2 are already complete. This plan begins at **Task 3** and covers the remaining streaming voice pipeline work:

1. safe streamed LLM output;
2. sentence assembly, single-sentence TTS, and bleat interstitials;
3. in-memory playback through a reusable output stream;
4. a single-producer streaming pipeline;
5. entrypoint integration;
6. regression, Raspberry Pi validation, documentation, and commits.

This workspace is a **development-only source environment**. Do not create, delete, copy, or rebuild a `.venv` here. Do not download models or run hardware-dependent checks here. Write tests now; run them later in the project’s configured runtime on the Raspberry Pi or CI.

## Architecture decisions

```text
LLM streamed chunks
→ visibility filter
→ sentence assembler
→ one producer thread synthesizes each completed sentence
→ queue: speech / optional interstitial / speech
→ main thread plays segments synchronously
```

- Sentence boundaries are `.`, `!`, and `?` only. Do not flush on commas or length limits.
- Decimal values such as `3.5` must not split, including when chunks divide the decimal.
- The single producer pauses LLM iteration while Kokoro synthesizes a sentence. This is intentional.
- Playback runs concurrently with the producer and must not block later LLM/TTS work.
- A bleat or silence interstitial appears only between two successfully synthesized speech segments—never first or last.
- No barge-in, echo cancellation, second LLM/TTS worker, or microphone capture while audio plays.

---

## Task 3 — Add a safe streaming LLM API

**Files**

- Modify: `src/chat_llm.py`
- Create: `tests/test_chat_llm_streaming.py`

**Interfaces**

```python
LLMChat.generate_response_chunks(user_prompt: str) -> Iterator[str]
LLMChat.generate_response(user_prompt: str) -> str
_StreamingResponseFilter.feed(chunk: str) -> str
_StreamingResponseFilter.finish() -> str
```

### Steps

1. Write failing tests using a fake streaming Llama backend. Cover:

   - visible chunks are yielded and history commits once after natural completion;
   - `<think>...</think>` is removed even when markers span chunks;
   - fenced code blocks are removed even when backticks span chunks;
   - `generate_response()` joins the streaming implementation;
   - closing a generator early does not commit partial history and clears thinking state.

2. Implement `_StreamingResponseFilter` with a small pending suffix buffer. It must track `inside_think` and `inside_code` state, recognizing:

   ```text
   <think>
   </think>
   ```
   ```
   ```
   ```

   Keep only a possible marker-prefix suffix between chunks; do not buffer the entire response merely to filter it.

3. Add stream-spacing normalization. Collapse repeated horizontal whitespace introduced by filtering, without altering punctuation or Vietnamese characters. Preserve a single boundary space when two visible chunks meet.

4. Implement `generate_response_chunks()` using:

   ```python
   self.llm.create_chat_completion(..., stream=True)
   ```

   Requirements:

   - set the thinking event for the generator lifetime;
   - extract `choices[0]["delta"]["content"]` safely;
   - filter and yield only visible text;
   - commit `(user_prompt, complete_reply)` to history only after natural exhaustion;
   - close the backend stream when possible;
   - clear the thinking event on normal completion, cancellation, and failure.

5. Make the existing batch API a wrapper:

   ```python
   def generate_response(self, user_prompt: str) -> str:
       return "".join(self.generate_response_chunks(user_prompt)).strip()
   ```

   Remove the old independent generation implementation after callers and tests use the new one.

6. In the configured test runtime, run:

   ```bash
   python -m pytest tests/test_chat_llm_streaming.py -q
   ```

   Then perform a real smoke test with one Vietnamese prompt. Confirm that thinking text and fenced code never appear in visible output and history remains correct.

7. Commit:

   ```bash
   git add src/chat_llm.py tests/test_chat_llm_streaming.py
   git commit -m "feat: stream cleaned LLM response chunks"
   ```

---

## Task 4 — Stateful sentence assembly, single-sentence TTS, and bleat interstitials

**Files**

- Modify: `src/voice_layer.py`
- Create: `tests/test_voice_layer_streaming.py`

**Interfaces**

```python
StreamingSentenceAssembler.feed(chunk: str) -> list[str]
StreamingSentenceAssembler.finish() -> str | None
split_sentences(text: str) -> list[str]
synthesize_sentence(tts, sentence: str) -> AudioSegment | None
load_bleat_segments(bleats_dir: Path) -> tuple[AudioSegment, ...]
build_inter_sentence_segment(bleats, *, probability=BLEAT_PROBABILITY, rng=random) -> AudioSegment
```

### Steps

1. Write failing assembler tests for:

   - sentence text arriving over multiple chunks;
   - `!` and `?` boundaries;
   - a decimal split across chunks (`"3."`, then `"5"`);
   - an unpunctuated final remainder;
   - `split_sentences()` producing the same results as the streaming assembler.

2. Implement `StreamingSentenceAssembler` with one mutable buffer.

   - Emit `!` and `?` immediately.
   - For `.`, wait if it follows a digit and no next character exists yet.
   - Treat a dot as decimal only when both adjacent characters are digits.
   - Normalize whitespace only when emitting a sentence or final tail.
   - Refactor `split_sentences()` to call `feed()` then `finish()`; delete the separate regex splitter.

3. Write failing TTS/bleat tests. Cover:

   - normal TTS output becomes mono, 16-bit, 24 kHz `AudioSegment`;
   - empty TTS waveform produces `None`;
   - an interstitial with a selected bleat contains pre-pause + bleat + post-pause;
   - missing/disabled bleats fall back to ordinary silence.

4. Extract `synthesize_sentence()` from the current batch synthesis body. It must call the existing loaded Kokoro adapter once, convert its NumPy waveform, normalize it, and return `None` for empty output. Make the existing `synthesize_sentences()` delegate to it.

5. Add `load_bleat_segments()`:

   - discover WAV files once at startup;
   - decode, normalize, fade, and apply configured gain once;
   - skip an invalid asset with a warning rather than failing startup;
   - return an immutable tuple.

6. Add `build_inter_sentence_segment()`:

   - use the configured probability and supplied RNG;
   - if no bleat is selected, return `SILENCE_MS` of normalized silence;
   - otherwise return pre-bleat silence + random loaded bleat + post-bleat silence.

7. Align the legacy batch compositor while it still exists: it must use the same loaded bleats/interstitial builder between successful speech segments, never append an interstitial after the last segment, and remove `compose_with_bleat()` if unused.

8. In the configured test runtime, run:

   ```bash
   python -m pytest tests/test_voice_layer_streaming.py -q
   ```

9. Commit:

   ```bash
   git add src/voice_layer.py tests/test_voice_layer_streaming.py
   git commit -m "feat: assemble streamed sentences and bleat gaps"
   ```

---

## Task 5 — Play in-memory segments through a reusable output stream

**Files**

- Modify: `src/audio_player.py`
- Create: `tests/test_audio_player.py`

**Interfaces**

```python
AudioPlayer.play_segment_blocking(segment: AudioSegment) -> None
AudioPlayer.play_blocking(audio_path: str | Path) -> None  # preserved
```

### Steps

1. Add fake-PyAudio tests. Verify that:

   - two valid in-memory segments reuse one output stream;
   - wrong frame rate/channel/sample width raises a clear `ValueError`;
   - `close()` closes the active stream and terminates PyAudio.

2. Add lazy stream state in `AudioPlayer`:

   ```python
   self._stream = None
   self._stream_format = None
   ```

   Use `(sample_width, channels, frame_rate, device_index)` as the stream-format key.

3. Implement `play_segment_blocking()`:

   - reject closed player instances;
   - require target mono/16-bit/24 kHz format;
   - obtain a matching stream using `_ensure_stream()`;
   - write raw bytes in `AUDIO_CHUNK_SIZE` frame chunks;
   - on output failure, close the stream and re-raise.

4. Implement `_ensure_stream()` and `_close_stream()`. Reuse a stream only while its format and selected output device match. `close()` must close the stream before terminating the backend.

5. Refactor compatibility `play_blocking()` to use the same stream writer rather than reopening/closing a stream for every file.

6. In the configured test runtime, run:

   ```bash
   python -m pytest tests/test_audio_player.py -q
   ```

7. Commit:

   ```bash
   git add src/audio_player.py tests/test_audio_player.py
   git commit -m "feat: play queued audio segments without reopening output"
   ```

---

## Task 6 — Build the single-producer streaming pipeline

**Files**

- Create: `src/streaming_response.py`
- Create: `tests/test_streaming_response.py`

**Interface**

```python
stream_response_to_player(
    *,
    prompt: str,
    llm,
    tts,
    player,
    bleat_segments: Sequence[AudioSegment],
    speech_end_time: float,
) -> bool
```

### Steps

1. Add ordering tests with fake LLM, TTS, and player:

   - `speech_1 → interstitial → speech_2`;
   - no leading/trailing interstitial;
   - an interstitial can be a real bleat or silence;
   - return `True` only when speech is played.

2. Add a concurrency test. The fake LLM must wait for an event set by playback of sentence 1 before yielding sentence 2. This proves playback starts before LLM generation has finished.

3. Add failure and empty-output tests:

   - producer-side LLM/TTS error is raised to the consumer/main thread;
   - empty response returns `False`;
   - worker thread is joined after all outcomes.

4. Implement an explicit queue protocol in `src/streaming_response.py`:

   ```python
   @dataclass(frozen=True)
   class _AudioItem:
       segment: AudioSegment
       kind: str  # "speech" or "interstitial"

   @dataclass(frozen=True)
   class _FailureItem:
       error: BaseException

   _END = object()
   ```

   Use an unbounded `queue.Queue()`: responses are limited to a few short sentences, and avoiding queue-full shutdown deadlocks is more valuable than an artificial bound.

5. Implement a single daemon producer thread:

   - consume `llm.generate_response_chunks(prompt)`;
   - feed chunks into `StreamingSentenceAssembler`;
   - synthesize each emitted sentence;
   - queue the first successful speech segment immediately;
   - for later successful speech segments, queue `build_inter_sentence_segment()` then speech;
   - process the final assembler tail;
   - close the LLM generator when available;
   - on exception queue `_FailureItem`;
   - always queue `_END` in `finally`.

6. Implement the consumer in `stream_response_to_player()`:

   - create queue, stop event, producer thread, and metrics;
   - consume queue items in the caller/main thread;
   - call `player.play_segment_blocking()` for every segment;
   - set first-playback timing when first audio is played;
   - on `_FailureItem`, set stop event and raise the original error;
   - in `finally`, set stop event and join the producer;
   - use a final 5-second defensive join timeout; if still alive, raise rather than starting another conversation cycle.

7. Add a private metrics dataclass and log:

   ```text
   speech end → first sentence boundary
   speech end → first synthesized segment
   speech end → first playback
   ```

8. In the configured test runtime, run:

   ```bash
   python -m pytest tests/test_streaming_response.py -q
   ```

9. Commit:

   ```bash
   git add src/streaming_response.py tests/test_streaming_response.py
   git commit -m "feat: stream sentences through TTS and playback"
   ```

---

## Task 7 — Integrate microphone selection and streaming into the voice entrypoint

**Files**

- Modify: `src/talking_sheep_voice.py`
- Modify: `tests/test_microphone_menu.py` if integration changes expectations
- Create or modify: `tests/test_talking_sheep_voice.py`

### Steps

1. Preserve completed microphone-selection behavior through a pure startup-policy helper:

   ```python
   resolve_startup_input_selector(
       recorder,
       configured_selector,
       *,
       interactive: bool,
       menu_selector=select_microphone_interactive,
   ) -> tuple[object | None, bool]
   ```

   Test configured-selector bypass, one-device auto-selection, non-interactive multiple-device fallback, and Escape cancellation.

2. In `main()`:

   - preserve `--list-mics` early exit before STT/LLM/TTS initialization;
   - use the completed microphone menu only when multiple physical inputs and an interactive terminal exist;
   - exit cleanly after Escape;
   - retain automatic recorder scoring under non-interactive startup;
   - keep explicit CLI/environment selection authoritative.

3. Write a `run_once()` integration test using fake recorder/STT and a monkeypatched `stream_response_to_player()`. Assert the streaming function receives exactly:

   ```python
   prompt, llm, tts, player, bleat_segments, speech_end_time
   ```

   The test must not provide or call `play_blocking(path)`, proving the old final-WAV path is gone.

4. Replace the normal batch response path. After a non-empty transcript, call:

   ```python
   completed = stream_response_to_player(
       prompt=transcript,
       llm=llm,
       tts=tts,
       player=player,
       bleat_segments=bleat_segments,
       speech_end_time=speech_end_time,
   )
   ```

   Keep existing microphone debug-audio saving in the outer `finally`.

5. Load bleats once after creating Kokoro and before the conversation loop:

   ```python
   bleat_segments = load_bleat_segments(bleats_dir)
   ```

   Pass the immutable tuple through the conversation loop. Do not rescan/decode bleats per response.

6. Update the module docstring to describe:

   ```text
   microphone → streaming Zipformer STT → streamed LLM chunks
   → sentence assembler → per-sentence Kokoro synthesis
   → queued speech/bleat segments → synchronous playback
   ```

   State that recording resumes only once playback and the producer are complete.

7. In the configured test runtime, run:

   ```bash
   python -m pytest \
     tests/test_microphone_menu.py \
     tests/test_talking_sheep_voice.py \
     tests/test_streaming_response.py -q
   ```

8. Commit:

   ```bash
   git add src/talking_sheep_voice.py tests/test_talking_sheep_voice.py
   git commit -m "feat: integrate startup mic menu and streamed speech"
   ```

---

## Task 8 — Full regression, Raspberry Pi validation, and documentation

**Files**

- Modify: `README.md`
- Verify: all source and test files from Tasks 3–7

### Steps

1. In the Raspberry Pi/CI runtime, run:

   ```bash
   python -m pytest -q
   python -m compileall -q src tests
   python -m src.talking_sheep_voice --help
   python -m src.talking_sheep_voice --list-mics
   ```

   `--list-mics` must exit without loading STT, LLM, or TTS.

2. Validate microphone startup behavior on the Pi:

   - one physical microphone auto-selects;
   - multiple microphones in SSH terminal show the arrow-key menu;
   - non-interactive service startup does not wait for input;
   - Escape exits before model initialization.

3. Validate a three-sentence streaming response:

   ```text
   sentence 1 → TTS 1 → queue/play speech 1
   sentence 2 → TTS 2 → queue/play interstitial → speech 2
   sentence 3 → TTS 3 → queue/play interstitial → speech 3
   end with no trailing interstitial
   ```

   Confirm sentence 1 plays before the final LLM sentence is generated.

4. Validate failure recovery:

   - no bleat assets: speech remains audible with silence gaps;
   - one invalid WAV: it logs and skips while valid assets load;
   - LLM/TTS/playback failure: no producer thread leaks into the next cycle.

5. Validate both Bluetooth system-default output and local output. Confirm the persistent PyAudio stream does not reopen between sentence/bleat segments.

6. Record at least five interactions and compare median speech-end-to-first-playback with the old batch pipeline. Acceptance: first playback occurs before complete response synthesis and is lower for multi-sentence answers.

7. Update `README.md` with:

   - one-device auto-selection, interactive menu, Escape, and non-interactive fallback;
   - streamed LLM → sentence TTS → playback behavior;
   - one-producer tradeoff: TTS pauses LLM generation, playback does not;
   - interstitial bleats controlled by `BLEAT_PROBABILITY`;
   - no barge-in yet and recording resumes only after playback;
   - normal streaming path no longer creates `runtime/final.wav`.

8. Final commit:

   ```bash
   git add README.md
   git commit -m "docs: explain streamed Talking Sheep pipeline"
   ```

---

## Acceptance criteria

- `generate_response_chunks()` yields only visible text and commits history once after successful exhaustion.
- Thinking and fenced-code content never reaches TTS.
- Sentence 1 is synthesized at its real boundary; decimals are not split.
- Playback of sentence 1 begins while the producer prepares later speech.
- Only one LLM/TTS producer exists; TTS blocking LLM iteration is intentional.
- Every successful speech-to-speech gap is an optional bleat or silence interstitial; there is no leading/trailing bleat.
- Missing/invalid bleats degrade to silence, not a failed interaction.
- Playback and producer failures cannot leak threads into the next conversation cycle.
- Microphone capture does not run during playback.
- Existing `LLMChat.generate_response()` and batch voice utilities remain compatible until their callers are migrated.
- Tests are written in this development workspace; they are executed only in the configured Raspberry Pi or CI runtime, where they must pass.

## Explicitly deferred

- Barge-in/user interruption while the sheep is speaking.
- Echo cancellation and simultaneous microphone/speaker operation.
- Separate concurrent LLM and TTS workers.
- Comma/token-limit forced sentence flushing.
- Dynamic Bluetooth sink switching while the application is running.
