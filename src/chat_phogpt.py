"""PhoGPT chat wrapper using a provisioned local GGUF model."""

from pathlib import Path
import importlib
import os
import threading
import shutil
from collections import deque

try:
    from .config import (
        MODEL_FILENAME,
        PHOGPT_CONTEXT,
        PHOGPT_HISTORY_MAXLEN,
        PHOGPT_MAX_TOKENS,
        PHOGPT_MODEL_REPO,
        PHOGPT_N_BATCH_MAX,
        PHOGPT_REPEAT_PENALTY,
        PHOGPT_TEMPERATURE,
        PHOGPT_TOP_P,
        PROMPT_TEMPLATE,
        SYSTEM_PROMPT,
    )
except ImportError:
    from src.config import (
        MODEL_FILENAME,
        PHOGPT_CONTEXT,
        PHOGPT_HISTORY_MAXLEN,
        PHOGPT_MAX_TOKENS,
        PHOGPT_MODEL_REPO,
        PHOGPT_N_BATCH_MAX,
        PHOGPT_REPEAT_PENALTY,
        PHOGPT_TEMPERATURE,
        PHOGPT_TOP_P,
        PROMPT_TEMPLATE,
        SYSTEM_PROMPT,
    )

_thinking_event = threading.Event()


def is_thinking() -> bool:
    """Return ``True`` while the LLM is generating a response."""
    return _thinking_event.is_set()


def load_runtime_dependencies():
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Install packages from requirements-rpi.txt before running PhoGPT."
        ) from exc

    return llama_cpp.Llama


def ensure_model(model_root: Path | None = None) -> Path:
    """Resolve the Q4 GGUF, downloading it once when it is not present."""
    configured = os.environ.get("PHOGPT_MODEL_PATH")
    if configured:
        model_path = Path(configured).expanduser()
    else:
        root = model_root or Path(__file__).resolve().parent.parent
        model_path = root / "models" / MODEL_FILENAME

    if not model_path.name.endswith("Q4_K_M.gguf"):
        raise ValueError(f"PHOGPT_MODEL_PATH must point to a Q4_K_M.gguf file, got {model_path.name}")

    model_path = model_path.resolve()
    if not model_path.is_file():
        if os.environ.get("PHOGPT_AUTO_DOWNLOAD", "1").lower() in {"0", "false", "no"}:
            raise FileNotFoundError(
                f"PhoGPT Q4 model not found at {model_path}. "
                "Set PHOGPT_AUTO_DOWNLOAD=1 to download it automatically."
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to download the PhoGPT Q4 model. "
                "Install requirements-rpi.txt and try again."
            ) from exc

        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"PhoGPT Q4 model not found. Downloading {MODEL_FILENAME} from {PHOGPT_MODEL_REPO}...")
        part_path = model_path.with_name(model_path.name + ".part")
        if part_path.exists():
            part_path.unlink()
        downloaded_path = hf_hub_download(
            repo_id=PHOGPT_MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=str(model_path.parent / ".hf-cache"),
        )
        shutil.copyfile(downloaded_path, part_path)
        part_path.replace(model_path)

    return model_path


def build_prompt(history, user_prompt: str) -> str:
    parts = [SYSTEM_PROMPT.strip()]
    for previous_user, previous_assistant in list(history)[-PHOGPT_HISTORY_MAXLEN:]:
        parts.append(
            f"### Câu hỏi: {previous_user.strip()}\n### Trả lời: {previous_assistant.strip()}"
        )
    parts.append(PROMPT_TEMPLATE.format(instruction=user_prompt.strip()))
    return "\n\n".join(parts)


class PhoGPTChat:
    """Reusable PhoGPT chat instance with bounded conversation history."""

    def __init__(self, model_root: Path | None = None):
        llama_class = load_runtime_dependencies()
        self.model_path = ensure_model(model_root)
        n_ctx = int(os.environ.get("PHOGPT_CONTEXT", str(PHOGPT_CONTEXT)))
        n_threads = int(os.environ.get("PHOGPT_THREADS", str(os.cpu_count() or 4)))
        self.llm = llama_class(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=min(PHOGPT_N_BATCH_MAX, n_ctx),
            verbose=False,
        )
        self.history: deque[tuple[str, str]] = deque(maxlen=PHOGPT_HISTORY_MAXLEN)

    def _generate_response_internal(self, user_prompt: str) -> str:
        prompt_text = build_prompt(self.history, user_prompt)
        output = self.llm(
            prompt_text,
            max_tokens=PHOGPT_MAX_TOKENS,
            temperature=PHOGPT_TEMPERATURE,
            top_p=PHOGPT_TOP_P,
            repeat_penalty=PHOGPT_REPEAT_PENALTY,
            stop=["### Câu hỏi:"],
        )
        reply = output["choices"][0]["text"].strip()
        if reply:
            self.history.append((user_prompt, reply))
        return reply

    def generate_response(self, user_prompt: str) -> str:
        """Generate a response and always clear the application status."""
        _thinking_event.set()
        try:
            return self._generate_response_internal(user_prompt)
        finally:
            _thinking_event.clear()


def main() -> None:
    llm = PhoGPTChat()
    print("PhoGPT> Xin chào! Bạn muốn trao đổi điều gì?")
    print("Nhập 'exit' để thoát.\n")
    while True:
        user_prompt = input("You> ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            break
        print(f"Assistant> {llm.generate_response(user_prompt)}\n")


if __name__ == "__main__":
    main()