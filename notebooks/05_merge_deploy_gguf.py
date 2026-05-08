# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load DPO model + merge adapter

# %%
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    except Exception:
        from transformers import AutoTokenizer
        ref = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        tokenizer.chat_template = ref.chat_template

# Phase 1: load SFT only — DPO applied after SFT is merged into base
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
model = PeftModel.from_pretrained(model, str(SFT_PATH))
print(f"Loaded SFT-mini adapter from {SFT_PATH}")

# %% [markdown]
# > **Note:** Unsloth `save_pretrained_merged` only supports a single PEFT layer.
# > We merge in 2 phases: first bake SFT into base, then stack DPO on the merged
# > base and merge again. This avoids a `NotImplementedError` from nested PeftModels.

# %% [markdown]
# ## 2. Save merged FP16 weights (2-phase: SFT → then DPO)

# %%
# 2-phase merge: SFT first, then DPO on top of merged base.
import gc, shutil

TEMP_SFT = REPO_ROOT / "adapters" / "merged-sft-temp"
TEMP_SFT.mkdir(parents=True, exist_ok=True)

# --- Phase 1: merge SFT into base ---
model.save_pretrained_merged(str(TEMP_SFT), tokenizer, save_method="merged_16bit")
print(f"Phase 1 done: base+SFT merged to {TEMP_SFT}")
del model; gc.collect(); torch.cuda.empty_cache()

# --- Phase 2: reload merged SFT, stack DPO, merge final ---
from unsloth import FastLanguageModel as _FLM
from peft import PeftModel as _PM

model, tokenizer = _FLM.from_pretrained(
    model_name=str(TEMP_SFT),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = _PM.from_pretrained(model, str(DPO_PATH))
print(f"Loaded DPO adapter from {DPO_PATH}")

model.save_pretrained_merged(str(MERGED_PATH), tokenizer, save_method="merged_16bit")
print(f"Phase 2 done: base+SFT+DPO merged to {MERGED_PATH}")
del model; gc.collect(); torch.cuda.empty_cache()

shutil.rmtree(str(TEMP_SFT), ignore_errors=True)
print(f"Saved merged FP16 (SFT+DPO) to {MERGED_PATH}")

# %% [markdown]
# ## 3. Quantize to GGUF Q4_K_M
#
# Q4_K_M is the sweet spot: ~4× compression vs FP16, minimal quality loss.
# Unsloth wraps llama.cpp's `quantize` binary — first run downloads + compiles
# llama.cpp (~3 min) then quantizes (~30 s).

# %%
# Reload the merged model — Unsloth's GGUF saver expects a live model handle.
from unsloth import FastLanguageModel as FLM

model, tokenizer = FLM.from_pretrained(
    model_name=str(MERGED_PATH),
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=False,    # already merged; load full precision
)

# %%
# Save GGUF in 1 quantization tier (Q4_K_M). Add more tiers below if you want the
# +3 "GGUF release published" rigor add-on.
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"Saved GGUF Q4_K_M to {GGUF_DIR}")

# %% [markdown]
# ### 3a. Additional quantization tiers (rigor add-on +3 "GGUF release published")
#
# Set `MULTI_QUANT=1` env to also export Q5_K_M and Q8_0 (~2× total disk).
# Each extra tier adds ~30s. Default ON when running the bonus track.

# %%
MULTI_QUANT = os.environ.get("MULTI_QUANT", "1") not in ("0", "false", "False", "")

if MULTI_QUANT:
    for q in ("q5_k_m", "q8_0"):
        try:
            model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method=q)
            print(f"  ✓ wrote {q}")
        except Exception as exc:
            print(f"  ✗ {q} failed: {exc}")
else:
    print("MULTI_QUANT=0 — only Q4_K_M exported. Set MULTI_QUANT=1 for +3 bonus.")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

del model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 6a. Push merged model + GGUF to HF Hub (Submission Option B = +5)
#
# Set `HF_TOKEN` and `HF_REPO_GGUF` env vars (or Colab Secrets). Pushes:
# - The merged FP16 weights (full HF model usable in vLLM/transformers)
# - All `*.gguf` files (Q4_K_M plus Q5_K_M / Q8_0 if MULTI_QUANT=1)
# - A model card with eval pointers + quantization table

# %%
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_GGUF = os.environ.get("HF_REPO_GGUF") or os.environ.get("HF_REPO")

# Optional pointers for the model card (read after NB3 ran)
dpo_metrics_path = REPO_ROOT / "adapters" / "dpo" / "dpo_metrics.json"
dpo_metrics = json.loads(dpo_metrics_path.read_text()) if dpo_metrics_path.exists() else {}

bench_path = REPO_ROOT / "data" / "eval" / "benchmark_results.json"
bench = json.loads(bench_path.read_text()) if bench_path.exists() else None

if HF_TOKEN and HF_REPO_GGUF:
    try:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        api.create_repo(HF_REPO_GGUF, exist_ok=True, private=False)

        # Build quant table from actual files on disk
        quant_rows = []
        for p in sorted(GGUF_DIR.glob("*.gguf")):
            quant = next((q for q in ("Q4_K_M", "Q5_K_M", "Q8_0", "q4_k_m", "q5_k_m", "q8_0") if q in p.name), p.stem)
            quant_rows.append((quant.upper(), p.name, p.stat().st_size / 1e6))
        quant_md = "| Quant | File | Size (MB) |\n|---|---|---:|\n"
        for q, name, mb in quant_rows:
            quant_md += f"| {q} | `{name}` | {mb:.1f} |\n"

        bench_md = ""
        if bench and "metrics" in bench:
            bench_md = "\n## Benchmark results (lab NB6)\n\n| Benchmark | SFT-only | SFT+DPO | Δ |\n|---|---:|---:|---:|\n"
            for b, scores in bench["metrics"].items():
                s, d = scores.get("sft", float("nan")), scores.get("dpo", float("nan"))
                delta = (d - s) if (s == s and d == d) else float("nan")
                bench_md += f"| {b} | {s:.3f} | {d:.3f} | {delta:+.3f} |\n"

        model_card = f"""---
base_model: {BASE_MODEL}
license: apache-2.0
language:
  - vi
  - en
datasets:
  - argilla/ultrafeedback-binarized-preferences-cleaned
  - bkai-foundation-models/vi-alpaca
tags:
  - dpo
  - alignment
  - gguf
  - llama-cpp
  - qwen2.5
  - vinuni-lab22
---

# Lab 22 DPO — merged + GGUF release ({HF_REPO_GGUF.split('/')[-1]})

Merged FP16 weights + GGUF quantizations of an SFT+DPO Qwen2.5 model trained
for the VinUni AICB Day 22 alignment lab.

## Pipeline

1. SFT-mini: 1k VN Alpaca · 1 epoch · LoRA r=16 on `{BASE_MODEL}`
2. DPO: 2k UltraFeedback · β={dpo_metrics.get('beta', 0.1)} · lr={dpo_metrics.get('lr', 5e-7)}
3. Merge SFT + DPO LoRA into base, save as FP16
4. Quantize via llama.cpp

## Available files

{quant_md}
- `model-*.safetensors` etc. — merged FP16 weights (vLLM / transformers)

## DPO training summary

| Metric | Value |
|---|---|
| Final training loss | {dpo_metrics.get('final_train_loss', 'n/a')} |
| End chosen reward | {dpo_metrics.get('end_chosen_reward', 'n/a')} |
| End rejected reward | {dpo_metrics.get('end_rejected_reward', 'n/a')} |
| End reward gap | {dpo_metrics.get('end_reward_gap', 'n/a')} |
{bench_md}

## Usage — llama-cpp-python (CPU/Metal/CUDA)

```python
from llama_cpp import Llama
llm = Llama(model_path="lab22-dpo-Q4_K_M.gguf", n_ctx={MAX_LEN}, n_gpu_layers=-1)
out = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Giải thích quicksort 3 câu."}}],
    max_tokens=200, temperature=0.0,
)
print(out["choices"][0]["message"]["content"])
```

## Usage — vLLM (BigGPU only)

```bash
vllm serve {HF_REPO_GGUF} --port 8000 --max-model-len {MAX_LEN}
```

## License & limitations

- Apache-2.0 (Qwen2.5 base).
- **Experimental** research model. Trained on English UltraFeedback;
  Vietnamese helpfulness/safety tested on 8 prompts (lab NB4).
- Not production-ready. Refusals on safety-critical prompts have not been
  exhaustively red-teamed.

## Citation

VinUni AICB program · Track 3 Day 22 · A20 cohort 2026.
"""
        (MERGED_PATH / "README.md").write_text(model_card, encoding="utf-8")

        # Upload merged FP16
        api.upload_folder(
            folder_path=str(MERGED_PATH),
            repo_id=HF_REPO_GGUF,
            commit_message="Lab 22 — merged FP16 weights + model card",
            ignore_patterns=["**/.cache/**", "**/__pycache__/**"],
        )
        # Upload all GGUF files
        for gguf_file in sorted(GGUF_DIR.glob("*.gguf")):
            api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=gguf_file.name,
                repo_id=HF_REPO_GGUF,
                commit_message=f"Add {gguf_file.name}",
            )
        print(f"✓ Pushed merged + {len(quant_rows)} GGUF file(s) to https://huggingface.co/{HF_REPO_GGUF}")
    except Exception as exc:
        print(f"HF push failed: {exc}  (no points lost — re-run later)")
else:
    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping HF push (Option B bonus +5).")
    elif not HF_REPO_GGUF:
        print("HF_REPO_GGUF / HF_REPO not set — skipping HF push.")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
