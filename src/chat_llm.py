"""Generic local GGUF chat wrapper."""

from collections import deque
from pathlib import Path
import importlib
import os
import re
import shutil
import threading

try:
    from .config import (
        LLM_CONTEXT,
        LLM_HISTORY_MAXLEN,
        LLM_MAX_TOKENS,
        LLM_MODEL_FILENAME,
        LLM_MODEL_REPO,
        LLM_N_BATCH_MAX,
        LLM_REPEAT_PENALTY,
        LLM_SYSTEM_PROMPT,
        LLM_TEMPERATURE,
        LLM_TOP_P,
    )
except ImportError:
    from src.config import (
        LLM_CONTEXT,
        LLM_HISTORY_MAXLEN,
        LLM_MAX_TOKENS,
        LLM_MODEL_FILENAME,
        LLM_MODEL_REPO,
        LLM_N_BATCH_MAX,
        LLM_REPEAT_PENALTY,
        LLM_SYSTEM_PROMPT,
        LLM_TEMPERATURE,
        LLM_TOP_P,
    )


_thinking_event = threading.Event()


def is_thinking() -> bool:
    return _thinking_event.is_set()


def load_runtime_dependencies():
    try:
        return importlib.import_module("llama_cpp").Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is missing. Install requirements-rpi.txt."
        ) from exc


def ensure_model(model_root: Path | None = None) -> Path:
    """Return the local GGUF path, downloading it once if missing."""
    configured = os.getenv("LLM_MODEL_PATH")

    if configured:
        model_path = Path(configured).expanduser()
    else:
        root = model_root or Path(__file__).resolve().parent.parent
        model_path = root / "models" / LLM_MODEL_FILENAME

    if not model_path.name.endswith("Q4_K_M.gguf"):
        raise ValueError(
            f"LLM_MODEL_PATH must point to a Q4_K_M GGUF file: {model_path.name}"
        )

    model_path = model_path.resolve()
    if model_path.is_file():
        return model_path

    if os.getenv("LLM_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
        raise FileNotFoundError(f"LLM model not found: {model_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download the model.") from exc

    model_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = model_path.with_suffix(model_path.suffix + ".part")
    part_path.unlink(missing_ok=True)

    print(
        f"Downloading {LLM_MODEL_FILENAME} "
        f"from {LLM_MODEL_REPO}..."
    )

    downloaded = hf_hub_download(
        repo_id=LLM_MODEL_REPO,
        filename=LLM_MODEL_FILENAME,
        local_dir=str(model_path.parent / ".hf-cache"),
    )

    shutil.copyfile(downloaded, part_path)
    part_path.replace(model_path)
    return model_path


def build_messages(
    history: deque[tuple[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": LLM_SYSTEM_PROMPT.strip(),
        }
    ]

    for user, assistant in history:
        messages.extend(
            [
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": assistant.strip()},
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": f"{user_prompt.strip()}\n/no_think",
        }
    )
    return messages


class LLMChat:
    """Reusable local LLM with bounded conversation history."""

    def __init__(self, model_root: Path | None = None):
        llama_class = load_runtime_dependencies()
        self.model_path = ensure_model(model_root)

        n_ctx = int(os.getenv("LLM_CONTEXT", str(LLM_CONTEXT)))
        n_threads = int(os.getenv("LLM_THREADS", str(os.cpu_count() or 4)))

        self.llm = llama_class(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=min(LLM_N_BATCH_MAX, n_ctx),
            verbose=False,
        )

        self.history: deque[tuple[str, str]] = deque(
            maxlen=LLM_HISTORY_MAXLEN
        )

    def _generate_response_internal(self, user_prompt: str) -> str:
        output = self.llm.create_chat_completion(
            messages=build_messages(self.history, user_prompt),
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repeat_penalty=LLM_REPEAT_PENALTY,
        )

       reply = output["choices"][0]["message"]["content"]

reply = re.sub(r"<think>.*?</think>\s*", "", reply, flags=re.DOTALL)
reply = re.sub(r"```.*?```", "", reply, flags=re.DOTALL).strip()

        if reply:
            self.history.append((user_prompt, reply))

        return reply

    def generate_response(self, user_prompt: str) -> str:
        _thinking_event.set()
        try:
            return self._generate_response_internal(user_prompt)
        finally:
            _thinking_event.clear()


def main() -> None:
    llm = LLMChat()
    print("Nhập 'exit' để thoát.\n")

    while True:
        user_prompt = input("You> ").strip()

        if user_prompt.lower() in {"exit", "quit"}:
            break

        if user_prompt:
            print(f"Assistant> {llm.generate_response(user_prompt)}\n")


if __name__ == "__main__":
    main()