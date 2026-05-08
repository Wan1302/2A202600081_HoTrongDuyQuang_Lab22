# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — DPO Training (the main event)
#
# **Stack:** TRL `DPOTrainer` + `DPOConfig(beta=0.1, lr=5e-7)` from deck §5.2.
# Maps to deck §3 (DPO derivation), §3.4 (failure modes — read closely!), §5.2 (TRL impl).
#
# > **Mục tiêu:** train DPO adapter on top of NB1 SFT-mini. Plot reward curves
# > (cả `chosen_rewards` và `rejected_rewards`). Save adapter to `adapters/dpo/`.
# >
# > Đây là **the** notebook quan trọng nhất của lab — 25/100 pts đến từ đây.
# > Đặc biệt là: **plot cả 2 curve riêng biệt**, không chỉ reward gap (deck §3.4).

# %% [markdown]
# ## 0. Setup

# %%
import os
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 4

# Hyperparameters from deck §5.2 lines 849–886
BETA = float(os.environ.get("DPO_BETA", "0.1"))
LR = float(os.environ.get("DPO_LR", "5e-7"))
EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_OUT = REPO_ROOT / "adapters" / "dpo"
PREF_PATH = REPO_ROOT / "data" / "pref" / "train.parquet"

DPO_OUT.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), f"NB1 must run first — {SFT_PATH} missing"
assert PREF_PATH.exists(), f"NB2 must run first — {PREF_PATH} missing"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"DPO hyperparams: beta={BETA}  lr={LR}  epochs={EPOCHS}")
print(f"max_length:      {MAX_LEN}  (prompt={MAX_PROMPT_LEN})")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"SFT input:       {SFT_PATH}")
print(f"output:          {DPO_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

# %% [markdown]
# ### 0a. Optional bonus: W&B login (rigor add-on +2)

# %%
USE_WANDB = bool(os.environ.get("WANDB_API_KEY"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "lab22-dpo")

if USE_WANDB:
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=WANDB_PROJECT,
            name=f"dpo-b{BETA}-{COMPUTE_TIER}",
            job_type="dpo",
            config={
                "tier": COMPUTE_TIER, "base_model": BASE_MODEL,
                "beta": BETA, "lr": LR, "epochs": EPOCHS,
                "max_length": MAX_LEN, "lora_r": 16, "lora_alpha": 32,
            },
            reinit=True,
        )
        print(f"W&B run initialized: {wandb.run.url if wandb.run else 'n/a'}")
    except Exception as exc:
        print(f"W&B init failed ({exc}) — falling back to report_to=none")
        USE_WANDB = False
else:
    print("WANDB_API_KEY not set — skipping W&B (no points lost).")

# %% [markdown]
# ## 1. Load policy + reference (the VRAM-doubling part)
#
# **Critical:** DPO needs the policy (trainable) AND a frozen reference (no grad).
# The reference is the SFT model at step 0; we load it twice. Unsloth's 4-bit base
# is shared across copies — only the LoRA adapter differs.

# %%
from unsloth import FastLanguageModel
from peft import PeftModel

# Policy — gets new DPO LoRA adapter on top of SFT LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load SFT adapter on top of base
model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")

# %%
# Wrap policy with NEW LoRA adapter for DPO updates (don't merge SFT — keep stacked)
# Unsloth re-applies LoRA on top of the existing PeftModel.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params (DPO LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# > **Why no separate `ref_model=` argument?** Modern TRL (≥ 0.12) auto-detects
# > PEFT models and uses the *base model without the adapter* as the reference.
# > That's the same memory layout: 1 base + 2 adapter sets in VRAM. No deepcopy
# > needed.

# %% [markdown]
# ## 2. Build DPOConfig (deck §5.2 hyperparameters)

# %%
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    beta=BETA,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    loss_type="sigmoid",         # DPO standard (alternatives: ipo, hinge, kto)
    report_to="wandb" if USE_WANDB else "none",
    run_name=f"dpo-b{BETA}-{COMPUTE_TIER}" if USE_WANDB else None,
)

print(f"DPOConfig: beta={dpo_config.beta}  lr={dpo_config.learning_rate}  loss_type={dpo_config.loss_type}")

# %% [markdown]
# ## 3. Load preference data

# %%
from datasets import Dataset

pref_ds = Dataset.from_parquet(str(PREF_PATH))
print(f"Loaded {len(pref_ds)} preference pairs from {PREF_PATH}")
print(f"Columns: {pref_ds.column_names}")

# %% [markdown]
# ## 4. Train

# %%
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,                # auto-derived from PEFT base
    args=dpo_config,
    train_dataset=pref_ds,
    tokenizer=tokenizer,
)

# %%
train_result = trainer.train()
print(f"\nFinal DPO loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## 5. Plot reward curves — THE diagnostic
#
# **Read deck §3.4 before interpreting these.** A growing reward gap can come from:
# - **(intended)** chosen reward going up + rejected staying flat
# - **(intended)** chosen rising slowly + rejected falling fast
# - **(likelihood displacement)** chosen reward going *down* + rejected falling faster
#
# The third case is what Razin et al. 2024 documented. It's not a bug, but it
# tells you the model is finding a way to widen the gap that doesn't necessarily
# improve actual chosen probability.

# %%
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame(trainer.state.log_history)
logs = logs[logs["loss"].notna() if "loss" in logs.columns else logs.index].copy()

# TRL DPO logs include rewards/chosen, rewards/rejected, rewards/margins, kl
chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

if chosen_col and rejected_col:
    axes[0].plot(logs["step"], logs[chosen_col], label="chosen reward", color="#2e548a", linewidth=1.5)
    axes[0].plot(logs["step"], logs[rejected_col], label="rejected reward", color="#c83538", linewidth=1.5)
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Implicit reward (log π/π_ref)")
    axes[0].set_title("Chosen vs Rejected rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    gap = logs[chosen_col] - logs[rejected_col]
    axes[1].plot(logs["step"], gap, color="#1a3355", linewidth=1.8)
    axes[1].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Reward gap (chosen − rejected)")
    axes[1].set_title("Reward gap (the headline number)")
    axes[1].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, "No reward columns in trainer.state.log_history.\nLikely TRL version mismatch.",
                 ha="center", va="center", transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, "—", ha="center", va="center", transform=axes[1].transAxes)

fig.suptitle(f"DPO reward curves · {COMPUTE_TIER} · β={BETA} · lr={LR}", y=1.02)
fig.tight_layout()

screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "03-dpo-reward-curves.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5a. Failure-mode self-check
#
# Read this cell carefully — it tells you which kind of "reward gap up" you got.

# %%
if chosen_col and rejected_col and len(logs) >= 5:
    last_chosen = logs[chosen_col].iloc[-5:].mean()
    last_rejected = logs[rejected_col].iloc[-5:].mean()
    last_gap = last_chosen - last_rejected
    first_chosen = logs[chosen_col].iloc[:5].mean()

    chosen_delta = last_chosen - first_chosen

    print(f"END  chosen reward:    {last_chosen:+.3f}")
    print(f"END  rejected reward:  {last_rejected:+.3f}")
    print(f"END  reward gap:       {last_gap:+.3f}")
    print()

    if last_gap < 0:
        print("✗ FAILURE: reward gap went NEGATIVE. DPO did the opposite of what you wanted.")
        print("  Likely causes: data quality (chosen/rejected swapped?), beta too high, lr too low.")
    elif chosen_delta < -0.5 and last_gap > 0:
        print("⚠  LIKELIHOOD DISPLACEMENT (deck §3.4):")
        print(f"   Reward gap is positive ({last_gap:+.3f}) — good!")
        print(f"   But chosen reward FELL by {chosen_delta:+.3f} during training.")
        print("   The gap grew because rejected fell faster than chosen.")
        print("   Document this in REFLECTION § 3 — it's a teachable moment, not a bug.")
    elif chosen_delta > 0 and last_gap > 0:
        print("✓ INTENDED: chosen reward UP and gap positive. Classic DPO success.")
    else:
        print("?  AMBIGUOUS: weak chosen movement + positive gap. Try longer training or higher lr.")

# %% [markdown]
# ## 6. Save adapter

# %%
trainer.model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))
print(f"Saved DPO adapter to {DPO_OUT}")

# Save the headline metrics for verify.py + REFLECTION
import json

metrics = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "beta": BETA,
    "lr": LR,
    "epochs": EPOCHS,
    "final_train_loss": float(train_result.training_loss),
    "end_chosen_reward": float(last_chosen) if chosen_col else None,
    "end_rejected_reward": float(last_rejected) if rejected_col else None,
    "end_reward_gap": float(last_gap) if chosen_col and rejected_col else None,
}
(DPO_OUT / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote metrics to {DPO_OUT / 'dpo_metrics.json'}")

if USE_WANDB:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            wandb.finish()
    except Exception as exc:
        print(f"W&B finish skipped: {exc}")

# %% [markdown]
# ## 6a. Optional bonus: push DPO adapter to HF Hub (Submission Option B = +5)
#
# Set `HF_TOKEN` and `HF_REPO` env vars (or Colab Secrets). If unset, this cell
# is a no-op — no points lost. The pushed repo gets a model card with: base
# model, dataset, hyperparameters, and reward gap.

# %%
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ADAPTER = os.environ.get("HF_REPO_ADAPTER", os.environ.get("HF_REPO"))

if HF_TOKEN and HF_REPO_ADAPTER and BETA == 0.1:
    try:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()

        model_card = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
datasets:
  - argilla/ultrafeedback-binarized-preferences-cleaned
language:
  - vi
  - en
tags:
  - dpo
  - alignment
  - lora
  - qwen2.5
  - vinuni-lab22
---

# Lab 22 DPO Adapter — {HF_REPO_ADAPTER.split('/')[-1]}

DPO LoRA adapter trained on top of an SFT-mini Qwen2.5 base for the VinUni AICB
Day 22 alignment lab. Stack: Unsloth + TRL `DPOTrainer`.

## Training details

| Field | Value |
|---|---|
| Base model | `{BASE_MODEL}` |
| Compute tier | {COMPUTE_TIER} |
| SFT predecessor | 1k VN Alpaca (`5CD-AI/Vietnamese-alpaca-cleaned`) |
| Preference dataset | `argilla/ultrafeedback-binarized-preferences-cleaned` (2k slice) |
| DPO β | {BETA} |
| DPO learning rate | {LR} |
| Epochs | {EPOCHS} |
| LoRA r / alpha | 16 / 32 |
| Max sequence length | {MAX_LEN} |

## Results

| Metric | Value |
|---|---|
| Final training loss | {metrics['final_train_loss']:.4f} |
| End chosen reward | {metrics['end_chosen_reward']:+.3f} |
| End rejected reward | {metrics['end_rejected_reward']:+.3f} |
| End reward gap | {metrics['end_reward_gap']:+.3f} |

## Usage

```python
from peft import PeftModel
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    "{BASE_MODEL}", load_in_4bit=True, max_seq_length={MAX_LEN}
)
model = PeftModel.from_pretrained(model, "{HF_REPO_ADAPTER}")
```

## License & limitations

- Base license: Apache-2.0 (Qwen2.5).
- This is an experimental research adapter. Not production-ready.
- Trained on English UltraFeedback; safety alignment tested on 8 VN prompts (see
  lab repo NB4). Use with care for deployment-critical workloads.

## Citation

VinUni AICB program · Track 3 Day 22 · A20 cohort 2026.
"""
        (DPO_OUT / "README.md").write_text(model_card, encoding="utf-8")

        api.create_repo(HF_REPO_ADAPTER, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(DPO_OUT),
            repo_id=HF_REPO_ADAPTER,
            commit_message=f"Lab 22 DPO adapter (β={BETA}, gap={metrics['end_reward_gap']:+.3f})",
        )
        print(f"✓ Pushed adapter + model card to https://huggingface.co/{HF_REPO_ADAPTER}")
    except Exception as exc:
        print(f"HF push failed: {exc}  (no points lost — re-run later or push via CLI)")
else:
    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping HF push (Option B bonus +5).")
    elif not HF_REPO_ADAPTER:
        print("HF_REPO / HF_REPO_ADAPTER not set — skipping HF push.")
    else:
        print(f"Skipping HF push for β={BETA} (only push the canonical β=0.1 run).")

# %% [markdown]
# ## 7. Vibe-coding callout
#
# Now's the time for the **β experiment** if you want the +6 rigor add-on.
#
# `make beta-sweep` runs this notebook 3 times with `DPO_BETA ∈ {0.05, 0.1, 0.5}`
# and saves to `adapters/dpo-b{0.05,0.1,0.5}/`. Plot the results yourself:
#
# ```python
# import json
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# results = []
# for d in sorted((REPO_ROOT / "adapters").glob("dpo-b*")):
#     m = json.loads((d / "dpo_metrics.json").read_text())
#     results.append((m["beta"], m["end_reward_gap"]))
# # plot β vs reward_gap
# ```
#
# **Think-hard zone:** what's the *expected* shape of the β-vs-reward-gap curve?
# Hypothesize before you look at the data. (Hint: deck §3.3.)
#
# **Next:** NB4 — qualitative side-by-side comparison.
