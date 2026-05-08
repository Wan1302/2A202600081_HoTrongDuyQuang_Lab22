#!/usr/bin/env python3
"""Stitch the per-stage notebooks/*.py files into a single Colab .ipynb.

Reads:
- notebooks/01_sft_mini.py  ← Stage 1
- notebooks/02_preference_data.py  ← Stage 2
- notebooks/03_dpo_train.py  ← Stage 3
- notebooks/03b_beta_sweep.py  ← Stage 3b (bonus)
- notebooks/04_compare_and_eval.py  ← Stage 4
- notebooks/05_merge_deploy_gguf.py  ← Stage 5
- notebooks/06_benchmark.py  ← Stage 6

Writes:
- colab/Lab22_DPO_T4.ipynb (single-file Colab notebook with Colab-specific
  prelude + login cells + all stages)
- colab/Lab22_DPO_BigGPU.ipynb (same but tier=BIGGPU)

Usage:
    python scripts/stitch_colab.py
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import jupytext

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "notebooks"
COLAB_DIR = REPO / "colab"

STAGES = [
    ("01_sft_mini.py", "Stage 1 — SFT-mini"),
    ("02_preference_data.py", "Stage 2 — Preference data"),
    ("03_dpo_train.py", "Stage 3 — DPO training"),
    ("03b_beta_sweep.py", "Stage 3b — β-sweep (bonus +6)"),
    ("04_compare_and_eval.py", "Stage 4 — Compare + cross-judge"),
    ("05_merge_deploy_gguf.py", "Stage 5 — Merge + GGUF + HF push"),
    ("06_benchmark.py", "Stage 6 — Benchmark (+ cross-judge AlpacaEval)"),
]


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def colab_prelude(tier: str) -> list[dict]:
    """Header + pip install + secrets + GPU probe + working dir cells."""
    cells = []

    cells.append(md_cell(
        f"""# Lab 22 — DPO/ORPO Alignment ({tier} tier)

**Track 3 · Day 22 · VinUni AICB program**

This is a single-file Colab notebook stitching all 6 stages of the lab + bonus add-ons:
1. SFT-mini build (replaces Lab 21)
2. Preference data prep
3. DPO training (the main event) — auto W&B + auto HF push
3b. β-sweep mini-experiment (bonus +6)
4. Side-by-side comparison + **cross-judge** (bonus +4)
5. Merge → GGUF (Q4_K_M + Q5_K_M + Q8_0) → HF release (bonus +5 + +3)
6. LLM benchmark (IFEval / GSM8K / MMLU / AlpacaEval-lite) + cross-judge

**Tier:** `{tier}` — {"Qwen2.5-3B + 2k UltraFeedback" if tier == "T4" else "Qwen2.5-7B + 5k UltraFeedback"}

> **Before running:** Runtime → Change runtime type → {"T4 GPU (free)" if tier == "T4" else "A100/L4 GPU"}. Verify with `nvidia-smi` cell below.
> **For bonuses:** open the 🔑 Secrets panel (left sidebar) and add:
> - `HF_TOKEN` (write-permission, https://huggingface.co/settings/tokens)
> - `HF_REPO` (e.g. `<your-username>/lab22-dpo-vn`)
> - `WANDB_API_KEY` (https://wandb.ai/authorize)
> - `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (for cross-judge +4)

> **Reference:** `README.md`, `COLAB-GUIDE.md`, `HARDWARE-GUIDE.md`,
> and the deck source `day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex`.
"""))

    cells.append(md_cell("## A. Colab setup — install deps + secrets + tier"))

    cells.append(code_cell(
        f"""# Set tier early — every downstream cell reads this.
import os
os.environ["COMPUTE_TIER"] = "{tier}"
print(f"COMPUTE_TIER set to {{os.environ['COMPUTE_TIER']}}")
"""))

    cells.append(code_cell(
        """# Install required packages (~3-5 min on Colab) — includes bonus deps
!pip install -q "unsloth>=2025.10" "trl>=0.12,<0.20" "peft>=0.13" "bitsandbytes>=0.44" \\
                "datasets>=3.1" "accelerate>=1.1" "llama-cpp-python>=0.3" \\
                "lm-eval[ifeval,math]>=0.4.5" \\
                "huggingface_hub>=0.26" "wandb>=0.18" \\
                matplotlib pandas pyarrow openai anthropic
"""))

    cells.append(code_cell(
        """# Pull bonus secrets from Colab userdata (🔑 panel, left sidebar)
# All are optional. Missing keys = bonus skipped, no points lost on core.
try:
    from google.colab import userdata  # type: ignore

    def _maybe(name):
        try:
            v = userdata.get(name)
        except Exception:
            v = None
        if v:
            os.environ[name] = v
            print(f"  ✓ {name} loaded into env")
        else:
            print(f"  – {name} not set (bonus skipped)")

    for k in ("HF_TOKEN", "HF_REPO", "HF_REPO_ADAPTER", "HF_REPO_GGUF",
             "WANDB_API_KEY", "WANDB_PROJECT",
             "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        _maybe(k)

    # If user only set HF_REPO, derive the two specific repo names
    if os.environ.get("HF_REPO"):
        if not os.environ.get("HF_REPO_ADAPTER"):
            os.environ["HF_REPO_ADAPTER"] = os.environ["HF_REPO"] + "-adapter"
        if not os.environ.get("HF_REPO_GGUF"):
            os.environ["HF_REPO_GGUF"] = os.environ["HF_REPO"] + "-gguf"
        print(f"  derived HF_REPO_ADAPTER={os.environ['HF_REPO_ADAPTER']}")
        print(f"  derived HF_REPO_GGUF   ={os.environ['HF_REPO_GGUF']}")
except ImportError:
    print("Not on Colab — set env vars manually (or use a .env file).")
"""))

    cells.append(code_cell(
        """# Probe GPU
import torch
assert torch.cuda.is_available(), "Enable GPU runtime: Runtime → Change runtime type → GPU"
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")
"""))

    cells.append(code_cell(
        """# Set up working directory matching the repo layout — Colab runs from /content
from pathlib import Path
WORK = Path("/content/lab22")
WORK.mkdir(exist_ok=True)
(WORK / "notebooks").mkdir(exist_ok=True)
(WORK / "scripts").mkdir(exist_ok=True)
(WORK / "data" / "pref").mkdir(parents=True, exist_ok=True)
(WORK / "data" / "eval").mkdir(parents=True, exist_ok=True)
(WORK / "adapters" / "sft-mini").mkdir(parents=True, exist_ok=True)
(WORK / "adapters" / "dpo").mkdir(parents=True, exist_ok=True)
(WORK / "adapters" / "merged-fp16").mkdir(parents=True, exist_ok=True)
(WORK / "gguf").mkdir(exist_ok=True)
(WORK / "submission" / "screenshots").mkdir(parents=True, exist_ok=True)

os.chdir(WORK / "notebooks")
print(f"Working dir: {Path.cwd()}")
"""))

    cells.append(code_cell(
        """# (Optional) Clone the lab repo so scripts/ are available for the β-sweep stage.
# If you uploaded the repo manually, comment out the clone line.
import subprocess, shutil

REPO_URL = "https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git"
LOCAL_REPO = Path("/content/Day22-Track3-DPO-Alignment-Lab")

if not LOCAL_REPO.exists() and "<your-username>" not in REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(LOCAL_REPO)], check=False)

# Copy scripts/ into WORK (so notebooks reference WORK/scripts/*)
if LOCAL_REPO.exists():
    for item in (LOCAL_REPO / "scripts").glob("*.py"):
        shutil.copy2(item, WORK / "scripts" / item.name)
    print(f"Copied scripts from {LOCAL_REPO / 'scripts'}")
else:
    print("⚠ Repo not cloned — β-sweep stage will need scripts/ uploaded manually.")
"""))

    cells.append(md_cell(
        """---
## Stages 1-6 stitched below
Each stage has its own header. Run cells in order. If you OOM, restart runtime
and reduce model size or batch (see `HARDWARE-GUIDE.md`).

> **Suggested run order:**
> 1. NB1 (~10 min) → NB2 (~2 min) → NB3 (~30 min)
> 2. *(Bonus +6)* NB3b β-sweep (~60 min — 2 extra DPO runs)
> 3. NB4 (~5 min) → NB5 (~5 min) → NB6 (~30 min)
> 4. Verify cell at the bottom + zip results
---
"""))
    return cells


def stage_cells(py_path: Path, label: str) -> list[dict]:
    """Convert a percent-format .py file into a list of nbformat cells."""
    nb = jupytext.read(str(py_path))
    cells = [md_cell(f"---\n# ⏵ {label}  ·  source: `notebooks/{py_path.name}`\n---\n")]
    for cell in nb.cells:
        # Convert NotebookNode → plain dict
        c: dict = {
            "cell_type": cell.cell_type,
            "id": _new_id(),
            "metadata": {},
        }
        # source can be str or list[str]
        src = cell.source
        if isinstance(src, str):
            src = src.splitlines(keepends=True) or [""]
        c["source"] = src
        if cell.cell_type == "code":
            c["execution_count"] = None
            c["outputs"] = []
        cells.append(c)
    return cells


def verify_cell() -> dict:
    return code_cell(
        """# Final verify — run pre-submission gatekeeper
import sys, subprocess
verify_script = LOCAL_REPO / "scripts" / "verify.py" if 'LOCAL_REPO' in dir() else WORK / "scripts" / "verify.py"
if verify_script.exists():
    subprocess.run([sys.executable, str(verify_script)], cwd=str(WORK))
else:
    print("verify.py not found — skipping. Run `python scripts/verify.py` after downloading results.")
"""
    )


def zip_results_cell() -> dict:
    return code_cell(
        """# Zip everything you need for REFLECTION
import shutil
result_zip = Path("/content/lab22_results.zip")
src_root = WORK
shutil.make_archive(
    str(result_zip).replace(".zip", ""),
    "zip",
    root_dir=str(src_root),
    base_dir=".",
)
print(f"Wrote {result_zip}  ({result_zip.stat().st_size/1e6:.1f} MB)")
print("Download via: Files → /content/lab22_results.zip → Right-click → Download")
"""
    )


def build(tier: str) -> dict:
    cells = colab_prelude(tier)
    for fname, label in STAGES:
        cells += stage_cells(NB_DIR / fname, label)
    cells.append(md_cell("---\n## Verify + zip results\n---\n"))
    cells.append(verify_cell())
    cells.append(zip_results_cell())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4" if tier == "T4" else "A100"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    COLAB_DIR.mkdir(exist_ok=True)
    for tier, fname in (("T4", "Lab22_DPO_T4.ipynb"), ("BIGGPU", "Lab22_DPO_BigGPU.ipynb")):
        nb = build(tier)
        out = COLAB_DIR / fname
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {out}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
