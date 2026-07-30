"""Wrapper chat GGUF local tổng quát."""

from collections import deque
from pathlib import Path
from typing import Iterator
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
        LLM_NUM_THREADS,
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
        LLM_NUM_THREADS,
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
    """Trả về đường dẫn GGUF local, tải xuống một lần nếu thiếu."""
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


class _StreamingResponseFilter:
    def __init__(self):
        self.inside_think = False
        self.inside_code = False
        self.buffer = ""

    def feed(self, chunk: str) -> str:
        output = []
        for char in chunk:
            candidate = self.buffer + char
            if self.inside_think:
                target = "</think>"
                longest_prefix = ""
                for i in range(len(candidate)):
                    suffix = candidate[i:]
                    if target.startswith(suffix):
                        longest_prefix = suffix
                        break
                self.buffer = longest_prefix
                if self.buffer == target:
                    self.inside_think = False
                    self.buffer = ""
            elif self.inside_code:
                target = "```"
                longest_prefix = ""
                for i in range(len(candidate)):
                    suffix = candidate[i:]
                    if target.startswith(suffix):
                        longest_prefix = suffix
                        break
                self.buffer = longest_prefix
                if self.buffer == target:
                    self.inside_code = False
                    self.buffer = ""
            else:
                targets = ["<think>", "```"]
                longest_prefix = ""
                for i in range(len(candidate)):
                    suffix = candidate[i:]
                    if any(t.startswith(suffix) for t in targets):
                        longest_prefix = suffix
                        break
                
                safe_len = len(candidate) - len(longest_prefix)
                if safe_len > 0:
                    output.append(candidate[:safe_len])
                    self.buffer = longest_prefix
                else:
                    self.buffer = candidate
                
                if self.buffer == "<think>":
                    self.inside_think = True
                    self.buffer = ""
                elif self.buffer == "```":
                    self.inside_code = True
                    self.buffer = ""
                    
        return "".join(output)

    def finish(self) -> str:
        if not self.inside_think and not self.inside_code:
            res = self.buffer
            self.buffer = ""
            return res
        self.buffer = ""
        return ""


class _SpacingNormalizer:
    def __init__(self):
        self.last_was_space = False

    def normalize(self, text: str) -> str:
        res = []
        for char in text:
            if char in (" ", "\t"):
                if not self.last_was_space:
                    res.append(" ")
                    self.last_was_space = True
            else:
                res.append(char)
                self.last_was_space = False
        return "".join(res)


class LLMChat:
    """LLM local có thể tái sử dụng với lịch sử hội thoại giới hạn."""

    def __init__(self, model_root: Path | None = None):
        llama_class = load_runtime_dependencies()
        self.model_path = ensure_model(model_root)

        n_ctx = int(os.getenv("LLM_CONTEXT", str(LLM_CONTEXT)))
        n_threads = LLM_NUM_THREADS

        self.llm = llama_class(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=min(LLM_N_BATCH_MAX, n_ctx),
            n_threads_batch=n_threads,
            verbose=False,
        )

        self.history: deque[tuple[str, str]] = deque(
            maxlen=LLM_HISTORY_MAXLEN
        )

    def generate_response_chunks(self, user_prompt: str) -> Iterator[str]:
        _thinking_event.set()
        stream = None
        filter_obj = _StreamingResponseFilter()
        normalizer = _SpacingNormalizer()
        complete_reply = []
        try:
            stream = self.llm.create_chat_completion(
                messages=build_messages(self.history, user_prompt),
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                repeat_penalty=LLM_REPEAT_PENALTY,
                stream=True,
            )
            for chunk in stream:
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    filtered = filter_obj.feed(content)
                    if filtered:
                        normalized = normalizer.normalize(filtered)
                        if normalized:
                            complete_reply.append(normalized)
                            yield normalized
            
            filtered_end = filter_obj.finish()
            if filtered_end:
                normalized_end = normalizer.normalize(filtered_end)
                if normalized_end:
                    complete_reply.append(normalized_end)
                    yield normalized_end
                    
            reply_str = "".join(complete_reply).strip()
            if reply_str:
                self.history.append((user_prompt, reply_str))
        finally:
            _thinking_event.clear()
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass

    def generate_response(self, user_prompt: str) -> str:
        return "".join(self.generate_response_chunks(user_prompt)).strip()


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