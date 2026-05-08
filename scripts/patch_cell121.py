import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "colab/Lab22_DPO_T4.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][121]["source"] = ["""\
# Fix: Transformers 5.5.0 revert_weight_conversion raises NotImplementedError
# for Unsloth-patched Qwen2.5. Guard flag prevents RecursionError on re-run.
try:
    import transformers.core_model_loading as _cml
    import transformers.modeling_utils as _mu
    if not getattr(_cml, "_revert_patched", False):
        _orig_revert = _cml.revert_weight_conversion
        def _safe_revert(model, state_dict):
            try:
                return _orig_revert(model, state_dict)
            except NotImplementedError:
                return state_dict
        _cml.revert_weight_conversion = _safe_revert
        _mu.revert_weight_conversion  = _safe_revert
        _cml._revert_patched = True
        print("Applied Transformers 5.5.0 compat patch")
    else:
        print("Compat patch already applied — skipping re-patch")
except Exception as _pe:
    print(f"Patch skipped: {_pe}")

_merged_done = any(MERGED_PATH.glob("*.safetensors"))
if _merged_done:
    print(f"Merged FP16 already in Drive ({MERGED_PATH}) — skipping merge.")
    import gc
    try:
        del model
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
else:
    import gc, shutil

    TEMP_SFT = REPO_ROOT / "adapters" / "merged-sft-temp"
    TEMP_SFT.mkdir(parents=True, exist_ok=True)

    # Phase 1: merge SFT into base
    model.save_pretrained_merged(str(TEMP_SFT), tokenizer, save_method="merged_16bit")
    print(f"Phase 1 done: base+SFT merged to {TEMP_SFT}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: reload merged SFT, stack DPO, merge final
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
    print("Saved merged FP16 (SFT+DPO) to", MERGED_PATH)
"""]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Patched cell 121 with re-run guard")
