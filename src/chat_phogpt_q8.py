from pathlib import Path
import importlib
import os


def load_runtime_dependencies():
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Install packages from requirements-rpi.txt before running PhoGPT."
        ) from exc

    return huggingface_hub.hf_hub_download, llama_cpp.Llama



MODEL_REPO = "vinai/PhoGPT-4B-Chat-gguf"
MODEL_FILENAME = "PhoGPT-4B-Chat-Q8_0.gguf"

SYSTEM_PROMPT = """Bạn là một chú cừu thân thiện, trả lời bằng tiếng Việt tự nhiên, rõ ràng và ngắn gọn.
Nếu câu hỏi thiếu ngữ cảnh, hãy hỏi lại để làm rõ thay vì đoán."""

PROMPT_TEMPLATE = "### Người dùng: {instruction}\n### Trả lời:"


def ensure_model(model_root: Path, hf_hub_download=None) -> Path:
    cache_dir = model_root / ".cache" / "phogpt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / MODEL_FILENAME

    if model_path.exists():
        return model_path

    if hf_hub_download is None:
        raise RuntimeError(
            f"Model not found at {model_path} and no download function available."
        )

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded_path)


def build_prompt(history: list[tuple[str, str]], user_prompt: str) -> str:
    parts = [SYSTEM_PROMPT.strip()]

    for previous_user, previous_assistant in history[-4:]:
        parts.append(
            f"### Câu hỏi: {previous_user.strip()}\n### Trả lời: {previous_assistant.strip()}"
        )

    parts.append(PROMPT_TEMPLATE.format(instruction=user_prompt.strip()))
    return "\n\n".join(parts)


class PhoGPTChat:
    """Wrapper for PhoGPT chat functionality."""
    
    def __init__(self, model_root: Path = None):
        hf_hub_download, Llama = load_runtime_dependencies()

        if model_root is None:
            model_root = Path(__file__).resolve().parent
        
        self.model_root = model_root
        self.model_path = ensure_model(model_root, hf_hub_download)
        n_ctx = int(os.environ.get("PHOGPT_CONTEXT", "4096"))
        n_threads = int(os.environ.get("PHOGPT_THREADS", str(os.cpu_count() or 4)))
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=min(512, n_ctx),
            verbose=False,
        )
        self.history: list[tuple[str, str]] = []
    
    def generate_response(self, user_prompt: str) -> str:
        """Generate response for user prompt."""
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


def main() -> None:
    hf_hub_download, Llama = load_runtime_dependencies()

    model_root = Path(__file__).resolve().parent
    model_path = ensure_model(model_root, hf_hub_download)
    n_ctx = int(os.environ.get("PHOGPT_CONTEXT", "4096"))
    n_threads = int(os.environ.get("PHOGPT_THREADS", str(os.cpu_count() or 4)))

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=min(512, n_ctx),
        verbose=False,
    )

    print("PhoGPT> Xin chào! Bạn muốn trao đổi điều gì?")
    print("Nhập 'exit' để thoát.\n")

    history: list[tuple[str, str]] = []

    while True:
        user_prompt = input("You> ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            break

        prompt_text = build_prompt(history, user_prompt)
        output = llm(
            prompt_text,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.05,
            stop=["### Câu hỏi:"],
        )

        reply = output["choices"][0]["text"].strip()
        history.append((user_prompt, reply))
        print(f"Assistant> {reply}\n")


if __name__ == "__main__":
    main()