# COLAB-GUIDE — Free T4 walkthrough cho Lab 22 (Option B + đủ bonus)

Hướng dẫn cụ thể từng bước chạy `colab/Lab22_DPO_T4.ipynb` trên **Free Colab T4** để đạt full điểm core (100) + đủ bonus (+20 cap).

---

## 1. Chuẩn bị — Setup tài khoản (làm 1 lần)

### 1.1 GitHub (bắt buộc)
1. Tạo repo public từ lab này. Nếu chưa có:
   ```bash
   gh repo create Day22-Track3-DPO-Alignment-Lab --public --source=. --push
   ```
   Hoặc trên web: GitHub → New repo → public → upload từ local.
2. Repo URL: `https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab`
3. Mở `README.md` line 27 — sửa `<your-username>` thành GitHub username thật để badge hoạt động.

### 1.2 HuggingFace token (Submission Option B = +5)
1. https://huggingface.co/settings/tokens → "New token" → **role = Write** → tên `lab22`.
2. Copy token (`hf_...`).
3. Tạo repo target trước (optional — code tự `create_repo` nếu chưa có):
   - Adapter: `https://huggingface.co/new` → name `lab22-dpo-vn-adapter` → **public**
   - GGUF: tương tự, name `lab22-dpo-vn-gguf` → **public**

### 1.3 Weights & Biases (+2 bonus)
1. https://wandb.ai/authorize → copy API key (40 ký tự).
2. (Optional) Tạo project name: `lab22-dpo` (mặc định trong code).

### 1.4 Judge API keys (+4 cross-judge bonus)
- **OpenAI:** https://platform.openai.com/api-keys (cần $5 credit, ~$0.02 cho 100 prompts × 2 judges)
- **Anthropic:** https://console.anthropic.com/settings/keys (cần $5 credit)
- Cả hai đều cho new account dùng free credit. Lab cần khoảng $0.10 tổng cộng.

---

## 2. Mở Colab + cấu hình runtime

### 2.1 Launch
1. Vào https://colab.research.google.com → **File → Open notebook → GitHub** → paste repo URL → chọn `colab/Lab22_DPO_T4.ipynb`.
2. **Hoặc** click trực tiếp badge "Open in Colab" trong README (sau khi sửa username ở bước 1.1).

### 2.2 Đổi runtime sang T4
1. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
2. **Connect** (góc phải trên) — chờ Colab cấp instance.

### 2.3 Add secrets
1. Click biểu tượng 🔑 (Secrets) ở sidebar trái.
2. Thêm từng secret bên dưới (toggle "Notebook access" cho mỗi cái):

| Name | Value | Bonus |
|---|---|---|
| `HF_TOKEN` | `hf_...` (từ bước 1.2) | +5 (Option B) |
| `HF_REPO` | `<your-hf-username>/lab22-dpo-vn` | (cùng +5) |
| `WANDB_API_KEY` | API key từ 1.3 | +2 |
| `OPENAI_API_KEY` | `sk-...` từ 1.4 | +4 (cross-judge) |
| `ANTHROPIC_API_KEY` | `sk-ant-...` từ 1.4 | (cùng +4) |

> Lab tự derive `HF_REPO_ADAPTER = HF_REPO + "-adapter"` và `HF_REPO_GGUF = HF_REPO + "-gguf"`. Hoặc bạn set trực tiếp 2 biến đó nếu muốn tên khác.

### 2.4 Clone repo vào Colab (cần cho NB3b β-sweep)
NB3b β-sweep gọi `scripts/train_dpo.py` — cần file đó tồn tại trong `/content/lab22/scripts/`. Có 2 cách:

**Cách A (tự động):** Sửa `REPO_URL` trong cell setup cuối Stage A của notebook, đổi `<your-username>` thành username GitHub của bạn. Cell tự `git clone` + copy scripts/.

**Cách B (manual):** Trong Colab Files panel (folder icon trái), upload thủ công 5 file `scripts/*.py` vào `/content/lab22/scripts/`.

> Nếu bạn skip β-sweep (mất +6 bonus), có thể bỏ qua bước 2.4.

---

## 3. Chạy pipeline — đúng thứ tự

### 3.1 Stage A — Setup (1 phút)
- Run các cell pip install, secrets, GPU probe, working dir.
- Verify GPU output: `Tesla T4  (16.0 GB)` hoặc tương đương.
- Verify secrets: thấy `✓` cho từng key đã add.

### 3.2 Stage 1 — NB1 SFT-mini (~10 phút)
- Run all cells trong stage 1.
- W&B link sẽ in ra ở cell init nếu setup đúng — **save link này**.
- Output cuối: `adapters/sft-mini/` + `submission/screenshots/02-sft-loss.png`.
- **Screenshot bắt buộc:**
  - `01-setup-gpu.png` — output cell GPU probe (chụp toàn cell)
  - `02-sft-loss.png` — đã tự lưu

### 3.3 Stage 2 — NB2 Preference data (~2 phút)
- Run all. Output: `data/pref/train.parquet` + `data/pref/eval.parquet`.

### 3.4 Stage 3 — NB3 DPO train (~30 phút)
- Đây là cell tốn thời gian nhất. Để Colab chạy, **không đóng tab**.
- W&B sẽ log loss + reward curves real-time → bạn xem được trên W&B web.
- Output cuối: `adapters/dpo/`, `dpo_metrics.json`, `03-dpo-reward-curves.png`.
- HF push tự chạy nếu set `HF_TOKEN` + `HF_REPO_ADAPTER` (chỉ push khi β=0.1).
- **Screenshot bắt buộc:** `03-dpo-reward-curves.png` (đã tự lưu).

### 3.5 Stage 3b — β-sweep (~60 phút, BONUS +6)
- Cần `/content/lab22/scripts/train_dpo.py` (xem 2.4).
- Re-run NB3 với β=0.05 (~30 min) + β=0.5 (~30 min). β=0.1 đã có sẵn.
- Output: `adapters/dpo-b0.05/`, `adapters/dpo-b0.5/`, `submission/screenshots/bonus-beta-sweep.png`.
- Skip nếu hết thời gian — **mất +6 nhưng không ảnh hưởng core**.

### 3.6 Stage 4 — NB4 Compare + cross-judge (~5 phút)
- Generate cho 8 prompts × 2 model = 16 outputs.
- Cross-judge tự chạy nếu có cả `OPENAI_API_KEY` + `ANTHROPIC_API_KEY`.
- Output: `04-side-by-side-table.png`, `judge_results.json`, `judge_cross.json` (nếu cross), `08-cross-judge-matrix.png` (nếu cross).
- **Screenshot bắt buộc:**
  - `04-side-by-side-table.png` (tự lưu)
  - `05-judge-output.png` — chụp output cell judge in ra (3+ verdicts có lý do)
- **Screenshot bonus:** `08-cross-judge-matrix.png` (tự lưu nếu cross-judge chạy).

### 3.7 Stage 5 — NB5 Merge + GGUF + HF push (~5–10 phút)
- Merge SFT + DPO LoRA → FP16 → quantize Q4_K_M + Q5_K_M + Q8_0.
- HF push tự chạy nếu có `HF_TOKEN` + `HF_REPO_GGUF` — push cả merged FP16 + 3 GGUF + model card.
- Output: `gguf/*.gguf` (3 files), `data/eval/deploy_meta.json`.
- **Screenshot bắt buộc:** `06-gguf-smoke.png` — chụp output cell smoke test (thấy filename + VN response).

### 3.8 Stage 6 — NB6 Benchmark (~30 phút)
- IFEval (~5 min) + GSM8K (~10 min) + MMLU sampled (~10 min) + AlpacaEval-lite (~5 min).
- Cross-judge AlpacaEval-lite tự chạy nếu có cả 2 keys.
- Output: `data/eval/benchmark_results.json`, `07-benchmark-comparison.png`.
- **Screenshot bắt buộc:** `07-benchmark-comparison.png` (tự lưu).

### 3.9 Verify + zip
- Cell `verify` — nên thấy "All checks passed" (trừ REFLECTION sẽ fail vì chưa fill).
- Cell zip — tạo `/content/lab22_results.zip`.
- Download zip về máy (Files panel → right-click).

---

## 4. Sau khi Colab xong — gửi tôi viết REFLECTION

Trong `lab22_results.zip` cần có (verify đã đảm bảo, nhưng bạn check thêm):

**Bắt buộc cho core 100 pts:**
- `adapters/sft-mini/adapter_config.json`
- `adapters/dpo/adapter_config.json` + `dpo_metrics.json`
- `data/pref/train.parquet`
- `data/eval/judge_results.json` + `side_by_side.jsonl`
- `data/eval/benchmark_results.json`
- `gguf/*Q4_K_M*.gguf`
- `submission/screenshots/02-sft-loss.png`
- `submission/screenshots/03-dpo-reward-curves.png`
- `submission/screenshots/04-side-by-side-table.png`
- `submission/screenshots/06-gguf-smoke.png`
- `submission/screenshots/07-benchmark-comparison.png`

**Bonus (mỗi cái cộng tương ứng):**
- W&B run URL (`+2`) — copy từ output cell W&B init
- HF model link `<user>/lab22-dpo-vn-adapter` (`+5` Option B)
- HF GGUF link `<user>/lab22-dpo-vn-gguf` với 3 quants (`+3`)
- `submission/screenshots/bonus-beta-sweep.png` + `data/eval/beta_sweep.json` (`+6`)
- `data/eval/judge_cross.json` + `08-cross-judge-matrix.png` + `alpaca_lite_cross_judge.json` (`+4`)

**Screenshots tôi cần thêm (chụp tay):**
- `01-setup-gpu.png` — output cell `nvidia-smi` hoặc GPU probe
- `05-judge-output.png` — output cell judge in ra 3+ verdicts có justification

Bạn upload `lab22_results.zip` (hoặc tar.gz) cho tôi → tôi sẽ:
1. Đọc `dpo_metrics.json`, `benchmark_results.json`, `judge_cross.json`, `beta_sweep.json`
2. Đọc số liệu thực + cross-reference deck §3.4 / §8.1
3. Viết `submission/REFLECTION.md` đầy đủ 7 sections (gồm § 5 β-sweep + § 7 alignment-tax)
4. Tick checkboxes Bonus tương ứng

---

## 5. Troubleshooting nhanh

| Triệu chứng | Fix |
|---|---|
| Cell W&B init lỗi `wandb.errors.UsageError: api_key` | `WANDB_API_KEY` chưa add vào Secrets, hoặc toggle "Notebook access" off. Re-add. |
| `huggingface_hub.errors.HfHubHTTPError: 401` | `HF_TOKEN` không phải Write permission. Tạo lại token với role=Write. |
| `OutOfMemoryError` ở DPO step 1 | Restart runtime. Verify `COMPUTE_TIER=T4`. Kiểm tra GPU = T4 (16 GB), không phải CPU. |
| β-sweep cell không tìm thấy `train_dpo.py` | Xem 2.4 — clone repo hoặc upload manual. |
| Cross-judge skipped | Set CẢ HAI `OPENAI_API_KEY` và `ANTHROPIC_API_KEY`. Chỉ 1 key → fallback single judge. |
| Colab disconnect giữa chừng | Free T4 cap ~12h/session. Chia làm 2 session: NB1-3 (1h) + NB3b-6 (2h). State persist via `/content/lab22/`. |
| GGUF push fail "file too large" | Chỉnh `MULTI_QUANT=0` trong Stage A — chỉ push Q4_K_M. Mất +3 nhưng giữ +5. |
| `lm_eval` IFEval crashes | Set env `HF_DATASETS_TRUST_REMOTE_CODE=1` ở đầu Stage 6 và rerun. |

---

## 6. Tổng thời gian + checkpoint

| Stage | Time | Checkpoint sau khi xong |
|---|---:|---|
| A. Setup | 5 min | GPU probe ✓ + 5 secrets ✓ |
| 1. SFT-mini | 10 min | `adapters/sft-mini/` |
| 2. Pref data | 2 min | `data/pref/train.parquet` |
| 3. DPO train | 30 min | `adapters/dpo/` + W&B link + HF adapter pushed |
| 3b. β-sweep | 60 min | `adapters/dpo-b{0.05,0.5}/` + plot |
| 4. Compare + judge | 5 min | `04-side-by-side` + cross-judge matrix |
| 5. Merge + GGUF | 8 min | 3 GGUF + HF gguf-repo pushed |
| 6. Benchmark | 30 min | `benchmark_results.json` + 4-bar plot |
| Zip + download | 2 min | `lab22_results.zip` về máy |
| **Tổng** | **~2.5h** | |

Free T4 session limit ~12h → an toàn rất nhiều. Nếu chia 2 session: dừng sau Stage 3, restart, mount Drive nếu muốn persist.

---

## 7. Sau khi tôi viết REFLECTION xong

1. Copy `REFLECTION.md` tôi viết vào `submission/REFLECTION.md` của repo local.
2. Add tất cả screenshots vào `submission/screenshots/`.
3. (Nếu chưa) `git add -A && git commit -m "Lab 22 submission"` + `git push`.
4. Verify repo public: mở https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab ở trình ẩn danh — phải xem được.
5. Paste public URL vào VinUni LMS Day-22 box. Done.

> Repo phải giữ public đến khi điểm được công bố. Private = 0 điểm.
