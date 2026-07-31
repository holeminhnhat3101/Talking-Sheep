import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, TypeAlias, Union

from pydub import AudioSegment

from src.voice_layer import (
    StreamingSentenceAssembler,
    build_inter_sentence_segment,
    synthesize_sentence,
    normalize_segment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SentenceMessage:
    text: str


@dataclass(frozen=True)
class _SentenceEnd:
    pass


@dataclass(frozen=True)
class _SentenceFailure:
    error: BaseException


@dataclass(frozen=True)
class _AudioSpeech:
    segment: AudioSegment


@dataclass(frozen=True)
class _AudioInterstitial:
    segment: AudioSegment


@dataclass(frozen=True)
class _AudioEnd:
    pass


@dataclass(frozen=True)
class _AudioFailure:
    error: BaseException


SentenceQueueItem: TypeAlias = Union[
    _SentenceMessage,
    _SentenceEnd,
    _SentenceFailure,
]

AudioQueueItem: TypeAlias = Union[
    _AudioSpeech,
    _AudioInterstitial,
    _AudioEnd,
    _AudioFailure,
]


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def _queue_put_until_stopped(
    q: queue.Queue,
    item,
    stop_event: threading.Event,
    timeout: float = 0.1,
) -> bool:
    """Queue an item with retries until stopped.
    
    Required control messages (_SentenceEnd, _SentenceFailure, _AudioEnd, _AudioFailure)
    must never be silently discarded. This preserves bounded backpressure while
    preventing loss of critical control messages.
    """
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue

    return False


def _queue_get_with_timeout(
    q: queue.Queue,
    stop_event: threading.Event,
    timeout: float = 1.0,
):
    """Return the next item or raise queue.Empty after the configured wait."""
    if stop_event.is_set():
        raise queue.Empty

    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        pass

    if stop_event.is_set():
        raise queue.Empty

    return q.get(timeout=0.1)


def _drain_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _StreamingMetrics:
    speech_end_time: float
    first_sentence_boundary: Optional[float] = None
    first_synthesized_segment: Optional[float] = None
    first_playback: Optional[float] = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def set_first_sentence_boundary(self, value: float) -> None:
        with self._lock:
            if self.first_sentence_boundary is None:
                self.first_sentence_boundary = value

    def set_first_synthesized_segment(self, value: float) -> None:
        with self._lock:
            if self.first_synthesized_segment is None:
                self.first_synthesized_segment = value

    def set_first_playback(self, value: float) -> None:
        with self._lock:
            if self.first_playback is None:
                self.first_playback = value

    def snapshot(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        with self._lock:
            return (
                self.first_sentence_boundary,
                self.first_synthesized_segment,
                self.first_playback,
            )


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _llm_worker(
    prompt: str,
    llm,
    sentence_queue: queue.Queue,
    stop_event: threading.Event,
    metrics: _StreamingMetrics,
) -> None:
    """Generate sentences continuously and place them on the TTS queue."""
    generator = None

    try:
        assembler = StreamingSentenceAssembler()
        generator = llm.generate_response_chunks(prompt)

        for chunk in generator:
            if stop_event.is_set():
                return

            for sentence in assembler.feed(chunk):
                if stop_event.is_set():
                    return

                sentence = sentence.strip()
                if not sentence:
                    continue

                logger.info("Assistant sentence: %s", sentence)
                metrics.set_first_sentence_boundary(time.monotonic())

                if not _queue_put_until_stopped(
                    sentence_queue,
                    _SentenceMessage(sentence),
                    stop_event,
                ):
                    return

        if stop_event.is_set():
            return

        tail = assembler.finish()
        if tail:
            tail = tail.strip()

        if tail:
            logger.info("Assistant sentence: %s", tail)
            metrics.set_first_sentence_boundary(time.monotonic())

            if not _queue_put_until_stopped(
                sentence_queue,
                _SentenceMessage(tail),
                stop_event,
            ):
                return

        _queue_put_until_stopped(
            sentence_queue,
            _SentenceEnd(),
            stop_event,
        )

    except BaseException as error:
        _queue_put_until_stopped(
            sentence_queue,
            _SentenceFailure(error),
            stop_event,
        )

    finally:
        if generator is not None:
            close = getattr(generator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug(
                        "Failed to close LLM generator.",
                        exc_info=True,
                    )


def _tts_worker(
    tts,
    bleat_segments: Sequence[AudioSegment],
    sentence_queue: queue.Queue,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    metrics: _StreamingMetrics,
) -> None:
    """Synthesize queued sentences and emit ordered audio segments."""
    first_sentence = True

    try:
        while not stop_event.is_set():
            try:
                item: SentenceQueueItem = _queue_get_with_timeout(
                    sentence_queue,
                    stop_event,
                )
            except queue.Empty:
                continue

            if isinstance(item, _SentenceFailure):
                _queue_put_until_stopped(
                    audio_queue,
                    _AudioFailure(item.error),
                    stop_event,
                )
                return

            if isinstance(item, _SentenceEnd):
                _queue_put_until_stopped(
                    audio_queue,
                    _AudioEnd(),
                    stop_event,
                )
                return

            if not isinstance(item, _SentenceMessage):
                raise TypeError(
                    f"Unsupported sentence queue item: {type(item).__name__}"
                )

            segment = synthesize_sentence(tts, item.text)
            if segment is None:
                continue

            # Normalize to target format (48kHz, stereo, 16-bit)
            segment = normalize_segment(segment)
            
            metrics.set_first_synthesized_segment(time.monotonic())

            if not first_sentence:
                interstitial = build_inter_sentence_segment(bleat_segments)
                if interstitial is not None:
                    # Normalize bleat to target format
                    interstitial = normalize_segment(interstitial)

                    if not _queue_put_until_stopped(
                        audio_queue,
                        _AudioInterstitial(interstitial),
                        stop_event,
                    ):
                        return

            if not _queue_put_until_stopped(
                audio_queue,
                _AudioSpeech(segment),
                stop_event,
            ):
                return

            first_sentence = False

    except BaseException as error:
        _queue_put_until_stopped(
            audio_queue,
            _AudioFailure(error),
            stop_event,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def stream_response_to_player(
    *,
    prompt: str,
    llm,
    tts,
    player,
    bleat_segments: Sequence[AudioSegment],
    speech_end_time: float,
) -> bool:
    """Run LLM, TTS, and playback as a bounded streaming pipeline."""
    sentence_queue = queue.Queue(maxsize=4)
    audio_queue = queue.Queue(maxsize=3)

    stop_event = threading.Event()
    metrics = _StreamingMetrics(speech_end_time=speech_end_time)

    llm_thread = threading.Thread(
        name="talking-sheep-llm",
        target=_llm_worker,
        args=(
            prompt,
            llm,
            sentence_queue,
            stop_event,
            metrics,
        ),
    )

    tts_thread = threading.Thread(
        name="talking-sheep-tts",
        target=_tts_worker,
        args=(
            tts,
            bleat_segments,
            sentence_queue,
            audio_queue,
            stop_event,
            metrics,
        ),
    )

    played_any = False
    pipeline_error: Optional[BaseException] = None

    llm_thread.start()
    tts_thread.start()

    try:
        # Start streaming session with persistent stream
        player.start_streaming_session()
        
        while True:
            try:
                item: AudioQueueItem = _queue_get_with_timeout(
                    audio_queue,
                    stop_event,
                    timeout=2.0,
                )
            except queue.Empty:
                if tts_thread.is_alive():
                    continue

                # TTS worker exited - check for final message
                try:
                    item = audio_queue.get_nowait()
                except queue.Empty:
                    raise RuntimeError(
                        "TTS worker exited without an audio terminal message."
                    )

            if isinstance(item, _AudioFailure):
                raise item.error

            if isinstance(item, _AudioEnd):
                break

            if isinstance(item, _AudioSpeech):
                metrics.set_first_playback(time.monotonic())
                player.write_pcm_blocking(item.segment.raw_data)
                played_any = True
                continue

            if isinstance(item, _AudioInterstitial):
                player.write_pcm_blocking(item.segment.raw_data)
                continue

            raise TypeError(
                f"Unsupported audio queue item: {type(item).__name__}"
            )

    except BaseException as error:
        pipeline_error = error
        raise

    finally:
        # End streaming session and drain PCM
        player.end_streaming_session()
        
        stop_event.set()

        llm_thread.join(timeout=5.0)
        tts_thread.join(timeout=2.0)

        alive_workers = [
            thread.name
            for thread in (llm_thread, tts_thread)
            if thread.is_alive()
        ]

        if alive_workers:
            termination_error = RuntimeError(
                "Pipeline workers failed to terminate cleanly: "
                + ", ".join(alive_workers)
            )

            if pipeline_error is None:
                raise termination_error

            logger.error("%s", termination_error)

        # Drain queues only after workers have exited
        _drain_queue(sentence_queue)
        _drain_queue(audio_queue)

        if played_any:
            (
                first_sentence_boundary,
                first_synthesized_segment,
                first_playback,
            ) = metrics.snapshot()

            if first_sentence_boundary is not None:
                logger.info(
                    "Speech end -> first sentence boundary: %.3f s",
                    first_sentence_boundary - metrics.speech_end_time,
                )

            if first_synthesized_segment is not None:
                logger.info(
                    "Speech end -> first synthesized segment: %.3f s",
                    first_synthesized_segment - metrics.speech_end_time,
                )

            if first_playback is not None:
                logger.info(
                    "Speech end -> first playback: %.3f s",
                    first_playback - metrics.speech_end_time,
                )

    return played_any