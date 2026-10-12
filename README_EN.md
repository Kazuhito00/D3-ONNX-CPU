[[Japanese](README.md)/[English](README_EN.md)]

# D3-ONNX-CPU

A minimal-dependency inference implementation of three of the Decision 3.0 models of the vLLM Semantic Router Team, [`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge) (0.59B), [`vllm-sr/d3-lite`](https://huggingface.co/vllm-sr/d3-lite) (0.85B) and [`vllm-sr/d3-nano`](https://huggingface.co/vllm-sr/d3-nano) (2.21B), on onnxruntime, tokenizers and numpy only (typed decisions over text and pictures). GPU inference is also possible.<br>
It does not use torch or transformers, which the vendor's runtime (`d3_runtime.py`) requires.

**d3-mini (5B) is not supported**: exporting its fp32 weights (about 18 GB, 4.54B parameters) needs about 25 GB of free RAM and was left out of this work (the machine has 31.7 GB shared with other jobs). See "Limitations".

# Features
- Minimal dependencies: onnxruntime, tokenizers and numpy at run time (plus pillow when pictures are used)
- Two scripts are all you need: `download_model.py` (fetch the models) and `demo_inference_text.py` (inference). Pictures: `demo_inference_image.py`
- Three sizes in one repository, one package `d3_decide` (`--model edge|lite|nano`)
- Decisions match the reference (the vendor runtime in PyTorch, backbone cast to fp32): d3-edge 110/110 answers (max probability difference 1.3e-5), d3-lite 110/110 (4.9e-6), d3-nano 64/64 (1.6e-5)
- Question types `choice`, `noul` (yes/no) and `score` (ordered levels), every answer with a probability for every option; nothing is generated
- Pictures are supported (any number per request; each is encoded on its own and the rows are concatenated), read with the vendor's rule (aspect ratio kept, area 65,536 to 1,638,400 pixels, one token per 32 x 32 pixels)
- The preprocessing (chat template, 3-D M-RoPE positions, picture patching and the two vision position tables) is ported to numpy and Pillow, with gates that compare it against transformers
- Runs on the CPU, and on the GPU (CUDA) with `--device cuda`. The providers actually used are reported; a silent fallback to the CPU is a warning (an error with `--require-gpu`)

# Purpose of This Repository
- How far the decisions of d3 can be run with 3 to 4 runtime dependencies
- Whether the vendor's preprocessing (prompt, tokenizer, picture processor, rope positions) can be ported and produce the same inputs
- What speed and memory look like on a CPU (and a small GPU)


# Requirements
```
Python 3.10 or later

onnxruntime  1.20 or later   (onnxruntime-gpu to run on the GPU; never both, see below)
tokenizers   0.20 or later
numpy        1.26 or later
pillow       10.0 or later   (only when pictures are used)
```
That is all that is needed to run (131 MB in a fresh `uv sync` environment; `import torch` fails there, which is the point). Downloading the models uses only the standard library.<br>
Environments are built with [uv](https://docs.astral.sh/uv/). Optional groups and extras:
- `cpu` (default group): onnxruntime (CPU build), installed by `uv sync`
- `gpu` (extra): onnxruntime-gpu; it cannot be installed together with `cpu` (see "Running on a GPU")
- `export`: torch, transformers 5.17.0, onnx and so on, to export the ONNX yourself or to run the oracle gates (never used at run time)

# Installation

```bash
git clone https://github.com/Kazuhito00/D3-ONNX-CPU
cd D3-ONNX-CPU

# runtime dependencies only (onnxruntime, tokenizers, numpy, pillow)
uv sync
```

# Download Model
The vendor publishes no ONNX. The exported ONNX files go on the [Releases](https://github.com/Kazuhito00/D3-ONNX-CPU/releases) page (they are not in the repository; the release has not been uploaded yet).<br>
The script below fetches them into `models/<model>/` with the standard library only (no git, no git-lfs).
```bash
uv run download_model.py                  # every model (edge, lite, nano: about 16 GB)
uv run download_model.py --model edge    # one model only (about 2.6 GB)
uv run download_model.py --model lite    # about 4.2 GB
uv run download_model.py --model nano    # about 9.3 GB
```
- With no arguments it fetches every model of the release (all graphs, the vision encoder, the tokenizer and the configuration). Use `--model` to take one
- `--seq-len 384` fetches only that bucket's graph (the weights, vision encoder and tokenizer always come along); `--exclude "vision_encoder*"` skips the vision encoder for text-only use
- Each model has its own `manifest-<model>.json` and assets named `d3-<model>__...`; every file is checked against the SHA-256 in the manifest (a mismatch deletes the file and stops)
- Files above 1.9 GiB are split into parts for GitHub's 2 GiB asset limit (the weights of lite: 2 parts, nano: 4 parts) and joined again after the download; nothing is fetched again when the files are already there, an interrupted download continues where it stopped

To export it yourself (downloads the original checkpoint from Hugging Face at a pinned revision, checks every file against the checkpoint's own `MODEL_MANIFEST.json`, writes the golden data with the vendor runtime, exports the ONNX; needs torch):
```bash
UV_PROJECT_ENVIRONMENT=.venv-export uv sync --group export      # keeps the torch environment apart from .venv
.venv-export/Scripts/python export/prepare_model.py --model edge --hf-dir <a folder outside the repository for the checkpoint>
.venv-export/Scripts/python export/prepare_model.py --model lite --seq-len 384 --seq-len 512      # only some buckets
```
The export of nano holds its 7.5 GB of fp32 weights in RAM (several times that while exporting; it ran on a 31.7 GB machine) and the golden run with the vendor runtime is slow (`--golden-args "--stride 2"` uses every second text request, which is what the shipped nano golden data does).

# Usage

### Example (Python)
```bash
uv run demo_inference_text.py                     # d3-edge
uv run demo_inference_text.py --model lite
uv run demo_inference_image.py --model nano
```
The picture demo uses `artifacts/images/sample_chart.png` by default (a bar chart: "which bar is the tallest?", "is this a bar chart?"). The sample is a synthetic picture drawn by a script, not a photograph. Use `--image` for your own pictures.
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
uv run demo_inference_image.py --max-pixels 50176      # cap the picture area (fewer tokens, faster, but not what the model was evaluated with)
```

```python
from d3_decide import D3Engine

engine = D3Engine("models/edge")            # picks the smallest bucket that holds each request

engine.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {"type": "choice", "instructions": "Which team should handle this request?",
                  "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                               "billing": "Payments, invoices and charges",
                               "technical": "Product setup and faults"}},
        "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
        "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]},
    },
    images=None,                            # or a list of paths, bytes, base64 data URLs or PIL images
)
# {"model": "d3-edge", "answers": {"route": {"type": "choice", "choice": "returns", "probabilities": {...}, "confidence": ...}, ...},
#  "usage": {"input_tokens": ..., "output_tokens": 0}}

engine.ask(state, images, "choice", "Which team?", [("billing", "charges"), ("technical", "bugs")])   # -> index of the best option
```
`system_one` has the same request and response shapes as the vendor's `model.system_one(state=..., questions=..., images=...)` (videos are not supported). A question that cannot be answered gets `{"type", "error", "message"}` with `invalid_question` or `max_length_exceeded` and the others are still answered. The model folder of each size is `models/<model>` (override: `--model-dir`, or `D3_EDGE_MODEL_DIR` / `D3_LITE_MODEL_DIR` / `D3_NANO_MODEL_DIR`).

### Question types
| `type` | Answer |
|---|---|
| `choice` | one of the options (`choice`, `confidence`, `probabilities`); `criteria` maps each option key to a description (or `None`) |
| `score` | the expected value on an ordinal scale (`score`, `legend`); 2 to 10 levels, the first is the lowest |
| `noul` | whether a statement holds (`noul` = probability of yes); `criteria` may describe `"false"` and `"true"` |

### Running on a GPU
`--device cuda` runs on CUDA (CUDAExecutionProvider first, CPU last); `--device auto` takes the first GPU provider the build has (CUDA, else DirectML); the default is `cpu`. The providers that really run are printed (`engine.providers_used` in Python). If the GPU cannot be used, a warning says so and the CPU runs instead; **`--require-gpu` turns that into an error**, which is what to use when a measurement must be a GPU measurement.
```bash
uv sync --extra gpu --no-group cpu       # onnxruntime-gpu instead of onnxruntime (they cannot coexist)
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
```
```python
engine = D3Engine("models/edge", device="cuda", require_gpu=True)
print(engine.providers_used)   # {"decoder": "CUDAExecutionProvider", "vision": "CUDAExecutionProvider"}
```
- Requirements (the combination used here): onnxruntime-gpu 1.31, CUDA 13.x and cuDNN 9 (CUDA 13.4 and cuDNN 9.20). `onnxruntime-gpu` does not bundle the CUDA DLLs: put the CUDA `bin\x64` and cuDNN `bin\13.2\x64` folders on `PATH` before starting (a Windows example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`, `C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64`). Without them onnxruntime drops the CUDA provider; that is what the warning / `--require-gpu` is for
- `onnxruntime` and `onnxruntime-gpu` write into the same directory: never keep both in one environment. The `cpu` group and the `gpu` extra exclude each other in `pyproject.toml`; use `uv run --no-sync` afterwards so that uv does not put the CPU build back (or use a separate environment: `uv venv .venv-gpu && uv pip install --python .venv-gpu onnxruntime-gpu tokenizers numpy pillow`)
- VRAM: the fp32 weights alone are 2.0 GB (edge), 3.0 GB (lite) and 7.5 GB (nano), plus the vision encoder; the graphs are loaded onto the GPU for each bucket. Only what was measured is stated below
- Measured on an RTX 3050 Ti Laptop (4 GB), `--require-gpu`, CUDA provider confirmed, 8 CPU threads, machine quiet (calibration 168-184 ms): **d3-edge is 4 to 6 times faster than on the CPU** (per question: L=128 75 ms vs 301 ms, L=384 144 ms vs 880 ms; one 224 px picture + 3 questions at L=256 349 ms vs 2,069 ms; vision encoder 21 ms vs 180 ms). **d3-lite is not faster** (L=128: 834 ms vs 769 ms per question; its 3 GB of weights nearly fill the 4 GB card, a part of the nodes stays on the CPU). **d3-nano on the GPU: not verified** (its 7.5 GB of weights do not fit in 4 GB; it was not tried). Answers on the GPU agree with the fp32 reference (d3-edge, 27 questions: same answer 27/27, max probability difference 7.7e-4; the CPU result is closer, 1.3e-5, because the order of operations differs). A bigger GPU was not measured
- onnxruntime prints a `ScatterND ... only guarantees to be correct if indices are not duplicated` warning for the CUDA provider; the indices of this graph are not duplicated

# How it works, and why the sequence length is fixed
**One question is one forward pass.** The prompt is a system turn ("You are a decision engine ..."), the user turn (pictures first, then `State:`, `Question:`, the options under single-token codes `A`, `B`, `C`, ... `AA`, ...) and an open assistant turn with an empty think block. The decoder returns the hidden state of the last prompt token, a 255 x hidden FP32 readout turns it into 255 code logits, and the answer is the softmax over the first N codes (N = number of options). A request with several questions runs them one after another; the pictures are encoded once.

The vendor lets the softmax-attention layers see the whole prompt (`noncausal_full_attention`: bidirectional, padding masked) while the Gated DeltaNet layers stay causal. The exported graph builds the same masks from `attention_mask`.

The Gated DeltaNet layers walk 64-token chunks in a Python loop, which the exporter **unrolls into a fixed number of steps**, so a graph with a dynamic sequence length cannot be built. The graph is exported at a fixed length L and the prompt is **right-padded** to L: the padded keys are masked in the softmax layers and the recurrent layers are causal, so the read position (the last real token) never sees the padding (`verify/check_buckets.py`: the same question through different buckets agrees to 2e-6). **The cost follows L, not the number of real tokens**, so six buckets are exported and the runtime loads the smallest one that holds the request:

| L | holds | use |
|---:|---|---|
| 128 | a short text question (about 90-125 tokens) | the smallest; fastest |
| 256 | a text question with a long state, or one 224 px picture (+64 tokens) and one question | |
| 384 | the default start of `--seq-len` | |
| 512 | a 640 x 480 picture (300 tokens) and a question | |
| 1024 | long states | |
| 2048 | a picture of up to about 1.6 MP (1,600 tokens) and a question | |

- `D3Engine(model_dir, seq_len=N)` starts on the smallest bucket >= N (default 384); a longer request switches to the smallest larger bucket that exists (the old graph is released first, so two decoders are never in memory) and stays there, so start on the bucket you expect (`--seq-len 128` for short questions) to avoid paying for a bigger one. A prompt beyond the largest bucket is answered `max_length_exceeded` (nothing is truncated)
- The buckets **share one weights file** (`models/<model>/onnx/weights.onnx.data`; tensors with identical bytes are read from the same place, so no number changes), so an extra bucket adds only its graph (58-64 MB for edge, about 165-185 MB for lite and nano)
- A 224 px picture is scaled to 256 x 256 (the vendor's minimum area) = 64 tokens; a 640 x 480 picture is 300 tokens

# Verification
```bash
uv run verify/run_all.py --model edge                       # every gate the environment can run (also lite, nano)
.venv-export/Scripts/python verify/run_all.py --model edge --hf-dir <checkpoint folder>   # with the oracle gates (torch + transformers 5.17.0)
```

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt | prompt string and token ids match the vendor runtime for all golden questions | runtime dependencies |
| verify | probabilities and answers vs the fp32 and bf16 reference (text, Japanese, JSON state, 2 to 30 options, one and two pictures) | runtime dependencies |
| check_buckets | the same question through different buckets, escalation to a larger bucket, `max_length_exceeded` | runtime dependencies |
| check_image | picture preprocessing vs the processor's own `pixel_values` (golden pictures) | + pillow |
| check_image_fuzz | the same on random pictures of 10 sizes (up to 3000 x 2000) | + transformers |
| check_tokenizer_fuzz | `tokenizers` vs transformers' tokenizer on 20,000 random mixed-script texts and prompts | + transformers |
| check_tables_torch | numpy vision position tables and M-RoPE vs transformers | + torch |
| tests/ | unit tests of questions, prompts, answers, picture geometry, provider selection (`python -m unittest discover -s tests`) | runtime dependencies |

`artifacts/<model>/golden.json` is ground truth made by actually running the **vendor runtime** (`export/make_golden.py`, CPU, transformers 5.17.0) twice: as shipped (bf16 backbone) and with the backbone cast to fp32. The ONNX graphs are fp32, so the fp32 run is the tight oracle and the bf16 run shows how far bf16 rounding itself moves the answer. The requests (`export/cases.py`) cover English and Japanese states, a JSON state, 2 to 30 options, choice / noul / score, one and two pictures.

| Model | questions | vs fp32 reference: same answer, max \|dp\| | vs bf16 (vendor as shipped): same answer, max \|dp\| |
|---|---:|---|---|
| d3-edge | 110 (39 requests) | 110/110, 1.25e-5 | 108/110, 3.3e-2 |
| d3-lite | 110 (39 requests) | 110/110, 4.9e-6 | 110/110, 3.7e-2 |
| d3-nano | 64 (23 requests: every second text request and all 7 picture requests) | 64/64, 1.6e-5 | 64/64, 3.6e-2 |

The two bf16 disagreements of d3-edge are cases where bf16 rounding itself moves the answer (it moves probabilities by up to 0.03-0.04 in all three models). Picture preprocessing: the largest difference to the processor's `pixel_values` is 2 grey levels (mean 1e-5) on random pictures of 10 sizes (`check_image_fuzz`), exactly 0 on the sample pictures; the vision position tables and M-RoPE agree exactly with transformers (40 grids, 60 prompt layouts); the tokenizer agrees on 30,000 random cases.

# Performance
CPU: Intel Core i7-12800H, 8 threads, onnxruntime 1.31, fp32; the GLiNER calibration of the benchmark (`runner.calibrate`; about 199 ms on a quiet machine) read 168-184 ms before and after each model, so the machine was quiet. Median of 5 runs after a warm-up; time per question including tokenization; "load" is the session creation of the graph.

| Model | L=128 | L=384 | L=512 | session load | peak RAM |
|---|---:|---:|---:|---:|---:|
| d3-edge (0.59B) | 301 ms | 880 ms | 1,187 ms | 8-9 s | 1.8-2.8 GB |
| d3-lite (0.85B) | 769 ms | 2,216 ms | 3,003 ms | 31-34 s | 3.3-5.1 GB |
| d3-nano (2.21B) | 1,263 ms | 3,786 ms | 5,059 ms | 41-64 s | 7.5-9.3 GB |

(per question, a text question of 108-128 tokens, 3 questions per request, run on the graph of the column.) With a picture: one 224 px picture (64 tokens) and 3 questions at L=256: edge 690 ms per question + 180 ms once for the vision encoder, lite 1,578 ms + 162 ms, nano 2,753 ms + 516 ms; a 640 x 480 picture (300 tokens) with 3 questions at L=512: edge 1,629 ms per question + 928 ms, lite 3,482 ms + 904 ms, nano 6,136 ms + 2,830 ms (peak RAM edge 3.2 GB, lite 5.5 GB, nano 10.0 GB). The cost follows the bucket, so start on the smallest bucket that holds your questions. GPU numbers: see "Running on a GPU".

# Limitations
- **d3-mini (5B) is not supported.** Exporting it needs about 25 GB of free RAM (about 18 GB of fp32 weights, plus the exporter's copies) and the vendor-runtime golden run in fp32 another 18 GB; it was left out on a 31.7 GB machine shared with other jobs. Its weights and the export code path are the same as the other three (`export/models.py` would only need an entry), but it was not tried, so nothing is claimed about it
- Videos are not supported (the vendor model reads them; this port does not). Pictures are read from paths, bytes, base64 data URLs or PIL images; URLs are not fetched
- The vendor's optional `permutation_average` is not implemented (it is off by default in the vendor runtime)
- The sequence length is fixed per graph, see above. Beyond the largest bucket (2048) a question is `max_length_exceeded`
- fp32 weights only. The vendor runs a bf16 backbone, so answers can differ near ties (see the table above). d3-nano's weights are 7.5 GB in fp32
- `tokenizer.json` says `[\p{L}\p{M}]+` in its pre-tokenizer but transformers 5.17.0's `Qwen2Tokenizer` (what the vendor runtime uses) replaces that regex by `\p{L}+`; this port uses the vendor's behaviour (`load_tokenizer`), which differs only for text with combining marks
- Do not export with `optimize=True` (the unrolled graphs make the optimizer very slow)
- The speed and accuracy figures of the vendor's model card were not reproduced here
- The fp32 golden data of d3-nano covers 64 of the 110 questions of the other two (every second text request); d3-mini has none

# Project Structure

```text
README.md / README_EN.md   # README (Japanese / English)
LICENSE                    # Apache-2.0
LICENSE-VLLM-SR            # the vendor's Apache-2.0 text, unchanged
NOTICE                     # attribution: this conversion + the vendors' NOTICE texts, kept as they are
pyproject.toml             # dependencies (run time: tokenizers, numpy, pillow + group cpu; extra gpu; group export)
download_model.py          # fetch the models (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py     # text inference demo
demo_inference_image.py    # picture inference demo
d3_decide/                 # the inference implementation (never imports export/)
  runtime.py               #   ORT sessions + buckets + right padding + readout + providers + system_one / ask
  format.py                #   questions, prompt, chat template, answers (port of d3_format.py and the question handling)
  chat.py                  #   3-D M-RoPE positions
  image.py                 #   Pillow decode + smart_resize + patching + the two vision position tables
  paths.py                 #   model folders
export/                    # heavy dependencies (torch / transformers / onnx)
  prepare_model.py         #   fetch the checkpoint + golden + export + share the weights
  export_onnx.py           #   decoder graph (one per L) + vision encoder
  graph_modules.py         #   the exported torch modules (also the fp32 oracle)
  share_external_data.py   #   buckets share one weights file (identical tensors only)
  make_golden.py, cases.py #   ground truth from the vendor runtime
  models.py                #   checkpoints and the pinned revisions
release/make_release.py    # builds the release files from models/ (split, SHA-256, manifest-<model>.json)
verify/                    # verification gates (run them all with run_all.py) and bench.py
tests/                     # unit tests
artifacts/<model>/         # golden.json (ground truth), verify_run.log; artifacts/images: sample pictures; artifacts/bench_smoke: benchmark smoke logs
models/<model>/            # ONNX, tokenizer and configs (gitignored)
```

# Reference
- [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) (revision `18d61df7`), [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) (`a8d25a91`), [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano) (`3aeb1c22`): the original models (vLLM Semantic Router Team)
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (base of d3-edge and d3-lite; d3-edge keeps 12 of its 24 layers), [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (base of d3-nano): Alibaba Cloud
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)

# License
D3-ONNX-CPU is under the [Apache-2.0 license](LICENSE).<br>
The model weights follow the license of the original models ([vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge), [d3-lite](https://huggingface.co/vllm-sr/d3-lite), [d3-nano](https://huggingface.co/vllm-sr/d3-nano): Apache-2.0, as their model cards, `LICENSE` files and Hugging Face tags say; the base models Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B are Apache-2.0 too). The vendor's `LICENSE` ([LICENSE-VLLM-SR](LICENSE-VLLM-SR)) and `NOTICE` texts are kept unchanged ([NOTICE](NOTICE) adds the note about this conversion and repeats the vendors' text); the release carries both next to the ONNX files of each model.

## About the converted files
The ONNX files in the release are a conversion of the checkpoints of [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge), [d3-lite](https://huggingface.co/vllm-sr/d3-lite) and [d3-nano](https://huggingface.co/vllm-sr/d3-nano) (vLLM Semantic Router Team; Apache-2.0, at the revisions above), made with this repository's `export/export_onnx.py` (the decoder and the vision encoder are exported as separate graphs, the decoder once per sequence length, with the weights shared between the lengths; the readout `readout.safetensors` is folded into the decoder graph; the masks of `noncausal_full_attention` are built inside the graph; the vision position tables are computed outside the graph). The weights are exported in fp32 (the original is bf16, which is an exact subset of fp32, so the weights themselves are unchanged). The conversion changes the numerics slightly (the vendor computes in bf16): see Verification. The providers of the original models do not endorse this conversion or warrant how it behaves.

The model cards do not describe the training data, and the evaluation figures in them are the vendor's own; they were not reproduced here. Read the original model cards before relying on the models.
