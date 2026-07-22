import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chat_phogpt


def test_thinking_state_success():
    llm = chat_phogpt.PhoGPTChat.__new__(chat_phogpt.PhoGPTChat)

    def generate(_prompt):
        assert chat_phogpt.is_thinking() is True
        return "Xin chào."

    llm._generate_response_internal = generate
    assert chat_phogpt.is_thinking() is False
    assert llm.generate_response("Chào bạn") == "Xin chào."
    assert chat_phogpt.is_thinking() is False


def test_thinking_state_failure():
    llm = chat_phogpt.PhoGPTChat.__new__(chat_phogpt.PhoGPTChat)

    def fail(_prompt):
        assert chat_phogpt.is_thinking() is True
        raise RuntimeError("generation failed")

    llm._generate_response_internal = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        llm.generate_response("Xin chào")
    assert chat_phogpt.is_thinking() is False


def test_q4_model_path_is_required_and_not_downloaded(tmp_path, monkeypatch):
    model = tmp_path / "PhoGPT-4B-Chat-Q4_K_M.gguf"
    model.write_bytes(b"test")
    monkeypatch.setenv("PHOGPT_MODEL_PATH", str(model))
    assert chat_phogpt.ensure_model() == model.resolve()


def test_non_q4_model_path_is_rejected(tmp_path, monkeypatch):
    model = tmp_path / "PhoGPT-4B-Chat.Q5_K_M.gguf"
    model.write_bytes(b"test")
    monkeypatch.setenv("PHOGPT_MODEL_PATH", str(model))
    with pytest.raises(ValueError, match="Q4_K_M"):
        chat_phogpt.ensure_model()


def test_missing_q4_model_is_downloaded(tmp_path, monkeypatch):
    model = tmp_path / "models" / "PhoGPT-4B-Chat-Q4_K_M.gguf"
    monkeypatch.setenv("PHOGPT_MODEL_PATH", str(model))

    def download(*, repo_id, filename, local_dir):
        assert repo_id == chat_phogpt.PHOGPT_MODEL_REPO
        assert filename == "PhoGPT-4B-Chat-Q4_K_M.gguf"
        downloaded = Path(local_dir) / filename
        downloaded.write_bytes(b"model")
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    assert chat_phogpt.ensure_model() == model.resolve()
    assert model.read_bytes() == b"model"