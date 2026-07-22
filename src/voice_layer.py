import re
import random
from pathlib import Path
from typing import Optional
from pydub import AudioSegment


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries."""
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def choose_bleat_position(
    sentences: list[str],
    probability: float = 0.3,
) -> Optional[int]:
    """Choose position to insert bleat (after sentence index)."""
    if len(sentences) < 2:
        return None
    if random.random() > probability:
        return None
    return 1


def choose_bleat(sentence: str) -> Path:
    """Select bleat file based on sentence ending punctuation."""
    if sentence.endswith("!"):
        return Path("assets/bleats/happy.wav")
    if sentence.endswith("?"):
        return Path("assets/bleats/confused.wav")
    return Path(
        random.choice([
            "assets/bleats/short.wav",
            "assets/bleats/happy.wav",
        ])
    )


def compose_response(
    sentence_files: list[Path],
    output_path: Path,
    bleat_position: Optional[int],
    bleat_path: Optional[Path],
) -> None:
    """Compose final WAV from sentence files with optional bleat insertion."""
    result = AudioSegment.empty()
    pause = AudioSegment.silent(duration=100)

    for index, sentence_file in enumerate(sentence_files):
        sentence_audio = AudioSegment.from_wav(sentence_file)
        result += sentence_audio

        if (
            bleat_position is not None
            and bleat_path is not None
            and index + 1 == bleat_position
        ):
            bleat = AudioSegment.from_wav(bleat_path)
            bleat = bleat.fade_in(25).fade_out(70)
            bleat = bleat - 3
            result += pause + bleat + pause

    result.export(output_path, format="wav")


def create_spoken_response(
    response_text: str,
    tts,
    runtime_dir: Path = Path("runtime"),
) -> Path:
    """Create spoken response with TTS and optional bleat insertion."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    
    sentences = split_sentences(response_text)

    if not sentences:
        raise ValueError("LLM returned no usable sentences")

    sentence_files: list[Path] = []

    for index, sentence in enumerate(sentences):
        output = runtime_dir / f"sentence_{index}.wav"
        if callable(tts):
            tts(sentence, str(output))
        elif hasattr(tts, "synthesize"):
            tts.synthesize(sentence, str(output))
        else:
            raise ValueError("TTS engine must be a callable or an object with a synthesize(text, output_path) method")
        sentence_files.append(output)

    bleat_position = choose_bleat_position(sentences)

    bleat_path = None
    if bleat_position is not None:
        preceding_sentence = sentences[bleat_position - 1]
        bleat_path = choose_bleat(preceding_sentence)

    final_path = runtime_dir / "final.wav"

    compose_response(
        sentence_files=sentence_files,
        output_path=final_path,
        bleat_position=bleat_position,
        bleat_path=bleat_path,
    )

    return final_path
