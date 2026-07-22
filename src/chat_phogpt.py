"""PhoGPT chat wrapper using a provisioned local GGUF model."""

from pathlib import Path
import importlib
import os
import threading

try:
    from .config import MODEL_FILENAME, PROMPT_TEMPLATE, SYSTEM_PROMPT
except ImportError:
    from src.config import MODEL_FILENAME, PROMPT_TEMPLATE, SYSTEM_PROMPT

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
    """Resolve the already-provisioned GGUF without downloading it."""
    configured = os.environ.get("PHOGPT_MODEL_PATH")
    if configured:
        model_path = Path(configured).expanduser()
    else:
        root = model_root or Path(__file__).resolve().parent.parent
        model_path = root / "models" / MODEL_FILENAME

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"PhoGPT Q4 model not found at {model_path}. "
            "Provision the model and set PHOGPT_MODEL_PATH; no model download is performed."
        )
    if not model_path.name.endswith("Q4_K_M.gguf"):
        raise ValueError(f"PHOGPT_MODEL_PATH must point to a Q4_K_M.gguf file, got {model_path.name}")
    return model_path


def build_prompt(history: list[tuple[str, str]], user_prompt: str) -> str:
    parts = [SYSTEM_PROMPT.strip()]
    for previous_user, previous_assistant in history[-4:]:
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
        n_ctx = int(os.environ.get("PHOGPT_CONTEXT", "2048"))
        n_threads = int(os.environ.get("PHOGPT_THREADS", str(os.cpu_count() or 4)))
        self.llm = llama_class(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=min(512, n_ctx),
            verbose=False,
        )
        self.history: list[tuple[str, str]] = []

    def _generate_response_internal(self, user_prompt: str) -> str:
        prompt_text = build_prompt(self.history, user_prompt)
        output = self.llm(
            prompt_text,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.05,
            stop=["### Câu hỏi:"],
        )
        reply = output["choices"][0]["text"].strip()
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