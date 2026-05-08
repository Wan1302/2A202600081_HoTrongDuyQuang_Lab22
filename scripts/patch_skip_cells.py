"""Patch Colab notebook cells with skip-if-exists conditions."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "colab/Lab22_DPO_T4.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def set_cell(idx, src):
    nb["cells"][idx]["source"] = [src]

# ── Cell 25: SFT train ──────────────────────────────────────────
set_cell(25, '''\
_sft_done = (ADAPTER_OUT / "adapter_config.json").exists()
if _sft_done:
    print(f"SFT adapter already in Drive ({ADAPTER_OUT}) — skipping training.")
    class _FakeResult:
        training_loss = float("nan")
    train_result = _FakeResult()
else:
    train_result = trainer.train()
    print(f"Final train loss: {train_result.training_loss:.4f}")
''')

# ── Cell 29: SFT save ───────────────────────────────────────────
set_cell(29, '''\
if _sft_done:
    print(f"Skipped save — SFT adapter already at {ADAPTER_OUT}")
else:
    trainer.model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))
    print(f"Saved SFT adapter to {ADAPTER_OUT}")
    if USE_WANDB:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"final_train_loss": float(train_result.training_loss)})
                wandb.finish()
        except Exception as exc:
            print(f"W&B finish skipped: {exc}")
''')

# ── Cell 47: preference data save ──────────────────────────────
old47 = "".join(nb["cells"][47].get("source", []))
indented47 = "\n".join("    " + l for l in old47.splitlines())
set_cell(47, f'''\
_pref_done = (PREF_OUT / "train.parquet").exists()
if _pref_done:
    print(f"Preference data already in Drive ({{PREF_OUT / 'train.parquet'}}) — skipping save.")
else:
{indented47}
''')

# ── Cell 66: DPO train ──────────────────────────────────────────
set_cell(66, '''\
_dpo_done = (DPO_OUT / "adapter_config.json").exists()
if _dpo_done:
    print(f"DPO adapter already in Drive ({DPO_OUT}) — skipping training.")
    class _FakeResult:
        training_loss = float("nan")
    train_result = _FakeResult()
else:
    train_result = trainer.train()
    print(f"Final DPO loss: {train_result.training_loss:.4f}")
''')

# ── Cell 72: DPO save + metrics ─────────────────────────────────
old72 = "".join(nb["cells"][72].get("source", []))
indented72 = "\n".join("    " + l for l in old72.splitlines())
set_cell(72, f'''\
if _dpo_done:
    print(f"Skipped save — DPO adapter already at {{DPO_OUT}}")
    _mp = DPO_OUT / "dpo_metrics.json"
    if _mp.exists():
        import json as _j
        metrics = _j.loads(_mp.read_text())
        last_chosen   = metrics.get("end_chosen_reward")
        last_rejected = metrics.get("end_rejected_reward")
        last_gap      = metrics.get("end_reward_gap")
        print(f"Loaded existing metrics — reward gap: {{last_gap}}")
else:
{indented72}
''')

# ── Cell 121: 2-phase merge ─────────────────────────────────────
old121 = "".join(nb["cells"][121].get("source", []))
indented121 = "\n".join("    " + l for l in old121.splitlines())
set_cell(121, f'''\
_merged_done = any(MERGED_PATH.glob("*.safetensors"))
if _merged_done:
    print(f"Merged FP16 already in Drive ({{MERGED_PATH}}) — skipping merge.")
    import gc
    try:
        del model
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
else:
{indented121}
''')

# ── Cell 123: load merged model for GGUF ────────────────────────
old123 = "".join(nb["cells"][123].get("source", []))
indented123 = "\n".join("    " + l for l in old123.splitlines())
set_cell(123, f'''\
import os
_all_quants = ["q4_k_m", "q5_k_m", "q8_0"] if os.environ.get("MULTI_QUANT", "1") not in ("0", "false", "False", "") else ["q4_k_m"]
_quants_missing = [
    q for q in _all_quants
    if not any(GGUF_DIR.glob(f"*{{q.upper()}}*.gguf")) and not any(GGUF_DIR.glob(f"*{{q}}*.gguf"))
]

if not _quants_missing:
    print(f"All GGUF files already in Drive ({{GGUF_DIR}}) — skipping model load.")
    model = None
    tokenizer = None
else:
    print(f"Missing GGUFs: {{_quants_missing}} — loading merged model...")
{indented123}
''')

# ── Cell 124: Q4_K_M export + push ─────────────────────────────
set_cell(124, '''\
import os

q4_file = next(iter(sorted(GGUF_DIR.glob("*[Qq]4_[Kk]_[Mm]*.gguf"))), None)
if q4_file:
    print(f"Q4_K_M already exists: {q4_file.name} — skipping export.")
elif model is None:
    print("Model not loaded (all GGUFs already done) — skipping.")
else:
    model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q4_k_m")
    q4_file = next(iter(sorted(GGUF_DIR.glob("*[Qq]4_[Kk]_[Mm]*.gguf"))), None)
    print(f"Saved Q4_K_M: {q4_file.name if q4_file else 'not found'}")

# Push to HF immediately (idempotent — safe to re-run)
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_GGUF = os.environ.get("HF_REPO_GGUF") or os.environ.get("HF_REPO")
if HF_TOKEN and HF_REPO_GGUF and q4_file:
    try:
        from huggingface_hub import HfApi, login
        login(token=HF_TOKEN, add_to_git_credential=False)
        _api = HfApi()
        _api.create_repo(HF_REPO_GGUF, exist_ok=True, private=False)
        _api.upload_file(
            path_or_fileobj=str(q4_file),
            path_in_repo=q4_file.name,
            repo_id=HF_REPO_GGUF,
            commit_message="Add Q4_K_M GGUF",
        )
        print(f"  ✓ Pushed {q4_file.name} → https://huggingface.co/{HF_REPO_GGUF}")
    except Exception as exc:
        print(f"  ✗ HF push Q4_K_M failed: {exc}")
else:
    print("  (HF_TOKEN / HF_REPO not set — Q4_K_M push skipped)")
''')

# ── Cell 126: Q5_K_M + Q8_0, skip if exists ────────────────────
set_cell(126, '''\
MULTI_QUANT = os.environ.get("MULTI_QUANT", "1") not in ("0", "false", "False", "")

if MULTI_QUANT:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_REPO_GGUF = os.environ.get("HF_REPO_GGUF") or os.environ.get("HF_REPO")

    for q in ("q5_k_m", "q8_0"):
        tag = q.upper()
        gguf_file = (
            next(iter(sorted(GGUF_DIR.glob(f"*{tag}*.gguf"))), None) or
            next(iter(sorted(GGUF_DIR.glob(f"*{q}*.gguf"))), None)
        )

        if gguf_file:
            print(f"  {tag} already exists: {gguf_file.name} — skipping export.")
        elif model is None:
            print(f"  Model not loaded — skipping {tag}.")
        else:
            try:
                model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method=q)
                gguf_file = (
                    next(iter(sorted(GGUF_DIR.glob(f"*{tag}*.gguf"))), None) or
                    next(iter(sorted(GGUF_DIR.glob(f"*{q}*.gguf"))), None)
                )
                print(f"  ✓ wrote {gguf_file.name if gguf_file else q}")
            except Exception as exc:
                print(f"  ✗ {tag} export failed: {exc}")
                print("    Previous tiers already pushed. Re-run this cell after restarting runtime.")
                break

        # Push immediately (idempotent)
        if HF_TOKEN and HF_REPO_GGUF and gguf_file:
            try:
                from huggingface_hub import HfApi, login
                login(token=HF_TOKEN, add_to_git_credential=False)
                _api = HfApi()
                _api.upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=gguf_file.name,
                    repo_id=HF_REPO_GGUF,
                    commit_message=f"Add {tag} GGUF",
                )
                print(f"    ✓ pushed {gguf_file.name} → HF")
            except Exception as push_exc:
                print(f"    ✗ push {tag} failed: {push_exc}")
else:
    print("MULTI_QUANT=0 — only Q4_K_M exported. Set MULTI_QUANT=1 for +3 bonus.")
''')

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Patched 8 cells with skip-if-exists conditions.")
print(f"Total cells: {len(nb['cells'])}")
