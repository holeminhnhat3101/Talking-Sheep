# Third-Party Licenses

This file lists third-party components that are directly used, downloaded, vendored, or modified by Talking Sheep.

It intentionally excludes ordinary transitive Python dependencies that are installed separately by `pip` and are not redistributed as part of this repository.

## Models and Model Artifacts

### Qwen3-1.7B GGUF

- **Component:** `ggml-org/Qwen3-1.7B-GGUF`
- **Upstream model:** `Qwen/Qwen3-1.7B`
- **Artifact used:** `Qwen3-1.7B-Q4_K_M.gguf`
- **License:** Apache License 2.0
- **Model page:** https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF
- **Upstream model page:** https://huggingface.co/Qwen/Qwen3-1.7B

The Qwen model weights and the GGUF conversion remain subject to their upstream license terms.

### Vietnamese Zipformer Streaming ASR

- **Component:** `hynt/Zipformer-30M-RNNT-Streaming-6000h`
- **Artifacts used:** encoder, decoder, joiner, token table, and BPE model
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
- **SPDX identifier:** `CC-BY-NC-ND-4.0`
- **Model page:** https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h
- **License text:** https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode

Important restrictions:

- Attribution is required.
- Commercial use is not permitted.
- Adapted or modified versions may not be distributed.

This checkpoint is not covered by the Apache License 2.0 used for the original Talking Sheep source code.

### Kokoro-Vietnamese

- **Component:** `contextboxai/Kokoro-Vietnamese`
- **Artifacts used:** ONNX model, PyTorch checkpoint where applicable, configuration, and Vietnamese voicepacks
- **License:** Apache License 2.0
- **Model page:** https://huggingface.co/contextboxai/Kokoro-Vietnamese
- **Source repository:** https://github.com/iamdinhthuan/Kokoro-Vietnamese
- **License text:** https://www.apache.org/licenses/LICENSE-2.0

Talking Sheep vendors and modifies parts of the Kokoro-Vietnamese source tree. Modified files should retain upstream notices and clearly indicate that changes were made.

## Direct Runtime Components

### llama-cpp-python

- **Component:** `llama-cpp-python`
- **License:** MIT License
- **Repository:** https://github.com/abetlen/llama-cpp-python
- **License file:** https://github.com/abetlen/llama-cpp-python/blob/main/LICENSE

### sherpa-onnx

- **Component:** `sherpa-onnx`
- **License:** Apache License 2.0
- **Repository:** https://github.com/k2-fsa/sherpa-onnx
- **License file:** https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE

### vig2p

- **Component:** `vig2p`
- **Package page:** https://pypi.org/project/vig2p/
- **License:** Not declared in the public PyPI metadata reviewed for the current release

Before redistributing `vig2p` as part of a complete device image, wheel bundle, or binary distribution, inspect the license files included with the exact installed version and record the result here.

To inspect the installed package metadata:

```bash
python - <<'PY'
from importlib.metadata import metadata

info = metadata("vig2p")
print("Version:", info.get("Version"))
print("License:", info.get("License"))
print("License-Expression:", info.get("License-Expression"))
print("Home-page:", info.get("Home-page"))
PY
```

### sea-g2p

- **Component:** `sea-g2p`
- **License:** Apache License 2.0
- **Package page:** https://pypi.org/project/sea-g2p/
- **Source repository:** https://github.com/pnnbao97/sea-g2p
- **License text:** https://www.apache.org/licenses/LICENSE-2.0

`sea-g2p` provides the G2P backend used by `vig2p`.

## Distribution Scope

For a source-only repository where users install dependencies themselves, this file covers the components that are most important to identify because they are:

- downloaded model weights,
- vendored or modified source code,
- or direct runtime engines central to the application.

If Talking Sheep is distributed as a complete Raspberry Pi image, executable bundle, appliance, or prebuilt Python environment, include the license texts and notices for every package and native library shipped in that bundle.

## Project License

The original Talking Sheep source code is licensed under the Apache License 2.0.

That project license does not replace or override any third-party license listed above.
