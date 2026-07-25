import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chat_llm


def test_thinking_state_success():
    llm = chat_llm.LLMChat.__new__(chat_llm.LLMChat)

    def generate(_prompt):
        assert chat_llm.is_thinking() is True
        return "Xin chào."

    llm._generate_response_internal = generate
    assert chat_llm.is_thinking() is False
    assert llm.generate_response("Chào bạn") == "Xin chào."
    assert chat_llm.is_thinking() is False


def test_thinking_state_failure():
    llm = chat_llm.LLMChat.__new__(chat_llm.LLMChat)

    def fail(_prompt):
        assert chat_llm.is_thinking() is True
        raise RuntimeError("generation failed")

    llm._generate_response_internal = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        llm.generate_response("Xin chào")
    assert chat_llm.is_thinking() is False


def test_q4_model_path_is_required_and_not_downloaded(tmp_path, monkeypatch):
    model = tmp_path / "Qwen3-1.7B-Q4_K_M.gguf"
    model.write_bytes(b"test")
    monkeypatch.setenv("LLM_MODEL_PATH", str(model))
    assert chat_llm.ensure_model() == model.resolve()


def test_non_q4_model_path_is_rejected(tmp_path, monkeypatch):
    model = tmp_path / "Qwen3-1.7B-Q5_K_M.gguf"
    model.write_bytes(b"test")
    monkeypatch.setenv("LLM_MODEL_PATH", str(model))
    with pytest.raises(ValueError, match="Q4_K_M"):
        chat_llm.ensure_model()


def test_missing_q4_model_is_downloaded(tmp_path, monkeypatch):
    model = tmp_path / "models" / "Qwen3-1.7B-Q4_K_M.gguf"
    monkeypatch.setenv("LLM_MODEL_PATH", str(model))

    def download(*, repo_id, filename, local_dir):
        assert repo_id == chat_llm.LLM_MODEL_REPO
        assert filename == "Qwen3-1.7B-Q4_K_M.gguf"
        downloaded = Path(local_dir) / filename
        downloaded.write_bytes(b"model")
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    assert chat_llm.ensure_model() == model.resolve()
    assert model.read_bytes() == b"model"
