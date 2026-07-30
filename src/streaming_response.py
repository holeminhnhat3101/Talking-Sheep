import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Sequence
from pydub import AudioSegment

from src.voice_layer import (
    StreamingSentenceAssembler,
    synthesize_sentence,
    build_inter_sentence_segment,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _AudioItem:
    segment: AudioSegment
    kind: str  # "speech" or "interstitial"

@dataclass(frozen=True)
class _FailureItem:
    error: BaseException

_END = object()

@dataclass
class _StreamingMetrics:
    speech_end_time: float
    first_sentence_boundary: Optional[float] = None
    first_synthesized_segment: Optional[float] = None
    first_playback: Optional[float] = None

def _producer_run(
    prompt: str,
    llm,
    tts,
    bleat_segments: Sequence[AudioSegment],
    q: queue.Queue,
    stop_event: threading.Event,
    metrics: _StreamingMetrics,
):
    try:
        assembler = StreamingSentenceAssembler()
        generator = llm.generate_response_chunks(prompt)
        first_speech = True
        
        for chunk in generator:
            if stop_event.is_set():
                break
            
            sentences = assembler.feed(chunk)
            for sentence in sentences:
                if stop_event.is_set():
                    break
                
                if metrics.first_sentence_boundary is None:
                    metrics.first_sentence_boundary = time.time()
                
                segment = synthesize_sentence(tts, sentence)
                if segment is None:
                    continue
                
                if metrics.first_synthesized_segment is None:
                    metrics.first_synthesized_segment = time.time()
                
                if first_speech:
                    first_speech = False
                    q.put(_AudioItem(segment, "speech"))
                else:
                    interstitial = build_inter_sentence_segment(bleat_segments)
                    q.put(_AudioItem(interstitial, "interstitial"))
                    q.put(_AudioItem(segment, "speech"))
        
        if not stop_event.is_set():
            tail = assembler.finish()
            if tail:
                if metrics.first_sentence_boundary is None:
                    metrics.first_sentence_boundary = time.time()
                
                segment = synthesize_sentence(tts, tail)
                if segment is not None:
                    if metrics.first_synthesized_segment is None:
                        metrics.first_synthesized_segment = time.time()
                    
                    if first_speech:
                        first_speech = False
                        q.put(_AudioItem(segment, "speech"))
                    else:
                        interstitial = build_inter_sentence_segment(bleat_segments)
                        q.put(_AudioItem(interstitial, "interstitial"))
                        q.put(_AudioItem(segment, "speech"))
                        
        if hasattr(generator, "close"):
            try:
                generator.close()
            except Exception:
                pass
                
    except BaseException as e:
        q.put(_FailureItem(e))
    finally:
        q.put(_END)

def stream_response_to_player(
    *,
    prompt: str,
    llm,
    tts,
    player,
    bleat_segments: Sequence[AudioSegment],
    speech_end_time: float,
) -> bool:
    q = queue.Queue()
    stop_event = threading.Event()
    metrics = _StreamingMetrics(speech_end_time=speech_end_time)
    
    producer = threading.Thread(
        target=_producer_run,
        args=(prompt, llm, tts, bleat_segments, q, stop_event, metrics),
        daemon=True,
    )
    
    played_any = False
    producer.start()
    
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            
            if isinstance(item, _FailureItem):
                stop_event.set()
                raise item.error
                
            if isinstance(item, _AudioItem):
                if metrics.first_playback is None and item.kind == "speech":
                    metrics.first_playback = time.time()
                
                player.play_segment_blocking(item.segment)
                if item.kind == "speech":
                    played_any = True
    finally:
        stop_event.set()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        
        producer.join(timeout=5.0)
        if producer.is_alive():
            logger.error("Producer thread did not terminate within 5 seconds.")
            raise RuntimeError("Producer thread did not terminate within 5 seconds")
            
        if played_any:
            if metrics.first_sentence_boundary is not None:
                logger.info("Speech end -> first sentence boundary: %.3f s", metrics.first_sentence_boundary - metrics.speech_end_time)
            if metrics.first_synthesized_segment is not None:
                logger.info("Speech end -> first synthesized segment: %.3f s", metrics.first_synthesized_segment - metrics.speech_end_time)
            if metrics.first_playback is not None:
                logger.info("Speech end -> first playback: %.3f s", metrics.first_playback - metrics.speech_end_time)
                
    return played_any
