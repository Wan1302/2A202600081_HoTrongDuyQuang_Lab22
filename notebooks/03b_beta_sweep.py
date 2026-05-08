# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3b — β-sweep mini-experiment (rigor add-on +6)
#
# **Run order:** NB1 → NB2 → NB3 (default β=0.1) → **NB3b** → NB4 → NB5 → NB6.
#
# > **Mục tiêu:** Re-train DPO với β ∈ {0.05, 0.5} (β=0.1 đã có sẵn từ NB3),
# > thu metrics, plot reward gap & chosen/rejected vs β, ghi `bonus-beta-sweep.png`.
# >
# > Trên T4 mỗi run mất ~30 min → tổng cộng thêm ~60 min cho NB3b.

# %% [markdown]
# ## 0. Setup

# %%
import os
import sys
import subprocess
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
PREF_PATH = REPO_ROOT / "data" / "pref" / "train.parquet"
DPO_DEFAULT = REPO_ROOT / "adapters" / "dpo"
SCREENSHOT_DIR = REPO_ROOT / "submission" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), "NB1 must run first"
assert PREF_PATH.exists(), "NB2 must run first"
assert DPO_DEFAULT.exists(), "NB3 must run first (β=0.1 baseline)"

# Sweep over the 2 *additional* betas (0.1 already done in NB3 → reuse it)
SWEEP_BETAS = [0.05, 0.5]
print(f"Sweep betas (additional): {SWEEP_BETAS}")
print(f"Will reuse existing β=0.1 from {DPO_DEFAULT}")

# %% [markdown]
# ## 1. Run β=0.05 and β=0.5
#
# Uses `scripts/train_dpo.py` (CLI wrapper that mirrors NB3 logic). Each run
# writes its adapter + `dpo_metrics.json` to `adapters/dpo-b{β}/`.

# %%
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_dpo.py"
assert TRAIN_SCRIPT.exists(), "scripts/train_dpo.py missing"

for beta in SWEEP_BETAS:
    out_dir = REPO_ROOT / "adapters" / f"dpo-b{beta}"
    if (out_dir / "dpo_metrics.json").exists():
        print(f"β={beta}: already done, skipping ({out_dir})")
        continue
    print(f"\n{'=' * 60}\nTraining β={beta}\n{'=' * 60}")
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--beta", str(beta),
        "--output-dir", str(out_dir),
        "--sft-path", str(SFT_PATH),
        "--pref-path", str(PREF_PATH),
    ]
    proc = subprocess.run(cmd, env=os.environ.copy())
    if proc.returncode != 0:
        print(f"⚠ β={beta} training failed (exit {proc.returncode}). Continuing with rest.")

# %% [markdown]
# ## 2. Symlink/copy the β=0.1 result into the sweep namespace
#
# `eval_judge.py --sweep-dir` scans `adapters/dpo-b*` — copy the canonical
# β=0.1 metrics over so the plot has 3 points.

# %%
import json
import shutil

CANON_OUT = REPO_ROOT / "adapters" / "dpo-b0.1"
if not CANON_OUT.exists():
    CANON_OUT.mkdir(parents=True, exist_ok=True)
    src_metrics = DPO_DEFAULT / "dpo_metrics.json"
    if src_metrics.exists():
        shutil.copy2(src_metrics, CANON_OUT / "dpo_metrics.json")
        print(f"Copied β=0.1 metrics to {CANON_OUT}")
    else:
        print(f"⚠ {src_metrics} missing — re-run NB3 first")

# %% [markdown]
# ## 3. Plot reward gap & chosen/rejected vs β
#
# Calls `scripts/eval_judge.py --sweep-dir adapters` which already has the plotting
# logic. Saves to `submission/screenshots/bonus-beta-sweep.png`.

# %%
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_judge.py"
PLOT_OUT = SCREENSHOT_DIR / "bonus-beta-sweep.png"

cmd = [
    sys.executable, str(EVAL_SCRIPT),
    "--sweep-dir", str(REPO_ROOT / "adapters"),
    "--output", str(PLOT_OUT),
]
subprocess.run(cmd, check=False)
print(f"\nPlot: {PLOT_OUT}")

# %% [markdown]
# ## 4. Print summary table

# %%
rows = []
for d in sorted((REPO_ROOT / "adapters").glob("dpo-b*")):
    m_path = d / "dpo_metrics.json"
    if m_path.exists():
        m = json.loads(m_path.read_text())
        rows.append(m)
rows.sort(key=lambda r: r.get("beta", 0))

print(f"{'β':>6}  {'final_loss':>11}  {'chosen':>9}  {'rejected':>9}  {'gap':>7}")
print("-" * 50)
for r in rows:
    print(
        f"{r.get('beta', 'n/a'):>6}  "
        f"{r.get('final_train_loss', float('nan')):>11.4f}  "
        f"{r.get('end_chosen_reward', float('nan')):>+9.3f}  "
        f"{r.get('end_rejected_reward', float('nan')):>+9.3f}  "
        f"{r.get('end_reward_gap', float('nan')):>+7.3f}"
    )

# Save as JSON for REFLECTION § 5
sweep_summary = REPO_ROOT / "data" / "eval" / "beta_sweep.json"
sweep_summary.parent.mkdir(parents=True, exist_ok=True)
sweep_summary.write_text(json.dumps(rows, indent=2))
print(f"\nSaved {sweep_summary}")

# %% [markdown]
# ## 5. Interpretation hints (deck §3.3)
#
# Theoretical expectation:
#
# - **Low β (0.05):** Less KL constraint to reference. Model moves further → larger reward
#   gap, but higher KL (model drifts more from SFT). Risk: weird outputs.
# - **Default β (0.1):** Sweet spot from deck §5.2.
# - **High β (0.5):** Strong KL constraint → conservative updates → smaller reward gap.
#
# **Your observation:** does the empirical gap match the prediction? If high β shows
# *larger* gap than low β, that's surprising — likely a sign of likelihood displacement
# under low β (deck §3.4). Document in REFLECTION § 5.
