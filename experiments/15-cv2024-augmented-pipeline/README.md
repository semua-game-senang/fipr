# exp015: FIPR critical path retrained IN-DOMAIN on Capsule Vision 2024's own train/val split

**Corrected 2026-08-17.** A first pass (see git history) trained the critical path on
Kvasir-Capsule only and evaluated zero-shot on the CV2024 test set — that was the WRONG
experiment for "compare directly against the challenge's literature": the 27 ranked teams and
6 organizer baselines in Table II/III of the challenge paper were all trained on CV2024's own
train/val pool, then scored on the CV2024 test set. Zero-shot cross-dataset generalization and
"comparable to this paper's own reported numbers" are two different claims. This restart trains
**in-domain**, on CV2024's own official train/validation split (10-class taxonomy: angioectasia,
bleeding, erosion, erythema, foreign body, lymphangiectasia, normal, polyp, ulcer, worms), for a
genuine apples-to-apples comparison against arXiv:2408.04940 Table II (27 teams) and Table III
(6 baselines). The discarded Kvasir-zero-shot artifacts are **not referenced** here.

## Scope

Teacher (DINOv3-S fine-tune) → alpha=0.75 width-interpolated KD student (NarrowResNet) →
structured pruning (PR+Fisher combined score, ratio=0.2, layer2/3/4) + finetune → low-rank SVD
(90% Frobenius energy, layer3/4) + finetune → INT8 PTQ (PR/width-informed granularity, CPU
fbgemm). Evaluated on CV2024's own official released test set (4,385 AIIMS frames), computing
Mean AUC and Balanced Accuracy exactly as the challenge does. CKA work and `paper/latex/*.tex`
are not touched by this experiment.

All new artifacts live under Modal volume path `/data/exp015_cv2024_indomain_pipeline/` and
under this directory's `checkpoints/`/`results/`. The previous pass's volume path
(`exp015_cv2024_augmented_pipeline/`) still exists on the volume but is **not read or written**
by any script here. The CV2024 official test-set images (already downloaded during the discarded
pass, at `/data/cv2024_test_exp015/`) ARE reused — that's just data, not a discarded result.

## 1. Dataset

- **Train + validation**: [figshare 26403469](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469)
  (`Dataset.zip`, 399 MB) — 10 class folders under `training/` and `validation/`, each with
  per-source-dataset subfolders (SEE-AI, KVASIR, KID, AIIMS). Downloaded and unpacked directly
  into the Modal volume by `training/stage0a_download_prep_data.py` (no local disk needed).
- **Test**: [figshare 27200664](https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664)
  — 4,385 AIIMS frames, class-labeled folders, already on the volume from the first pass.
- **Official split boundary used as-is** — not re-split. Confirmed counts match the challenge
  paper exactly: **37,607 train / 16,132 validation / 4,385 test** images (Table I of
  arXiv:2408.04940).
- **Leakage sanity check** (informational only — the official boundary is never overridden):
  171 exact-duplicate images (by filename + content hash) found shared across the official
  train/val boundary out of ~53,700 combined images (0.3%). See
  `results/stage0a` output / `stage0a_download_prep_data.py` docstring for methodology (fast
  filename-overlap-first, hash-only-the-overlap approach — hashing the full ~53k-image pool
  directly over the Modal volume network filesystem was tried first and was too slow).

## 2. Augmentation policy

In-domain training means the Kvasir→CV2024 domain-gap rationale from the discarded first pass no
longer applies. Kept it deliberately simple (this isn't the load-bearing design decision this
time — in-domain training is): horizontal + vertical flip (p=0.5 each, no canonical VCE frame
orientation), mild rotation (±10°), and brightness/contrast/saturation jitter (±20%/±20%/±15%).
Applied identically across stage0 (teacher), stage1 (KD student), stage3 (pruned finetune), and
stage4 (low-rank finetune). No blur/noise (present in the discarded pass's aggressive
domain-gap-driven policy — dropped here as unnecessary).

For reference, actual CV2024 teams' augmentation choices (from Annexure A, arXiv:2408.04940):
flips (~12 of 27 teams), rotation (~8), brightness/photometric jitter (~5, including rank-1
PuppyOps), zoom/crop (~4), Gaussian noise (~3), MixUp/SMOTE (1 each, imbalance-focused). Our
policy (flips + mild rotation + photometric jitter) sits squarely in the mainstream of what
placed teams used.

## 2a. Note on this package's checkpoints

This packaged copy includes every checkpoint on the FIPR critical path from stage 1 (KD
student) onward — the ones actually relevant to the compression story (~5.7 MB total). The
stage-0 **teacher checkpoint** (DINOv3-S fine-tune, ~86 MB) is intentionally **excluded** from
this package to keep the repository lightweight and git-friendly; it is not part of the
deployed/compressed pipeline, only an intermediate distillation source. It remains available on
the Modal volume at `/data/exp015_cv2024_indomain_pipeline/stage0/dinov3_teacher_cv2024_indomain.pt`
and can be regenerated by re-running `training/stage0_train_teacher.py`.

## 3. Pipeline stages and scripts

| Stage | Script | What it does |
|---|---|---|
| 0a | `training/stage0a_download_prep_data.py` | Download+unpack CV2024 train/val, build manifests, leakage sanity check |
| 0 | `training/stage0_train_teacher.py` | DINOv3-S fine-tune, 10-class head, CV2024 train/val |
| 1 | `training/stage1_kd_alpha075.py` | alpha=0.75 width-interpolated KD student (NarrowResNet) |
| 2 | `training/stage2_prune_surgery.py` | PR+Fisher structured pruning surgery (ratio=0.2), no finetune |
| 3 | `training/stage3_finetune_pruned.py` | KD finetune of the pruned checkpoint |
| 4 | `training/stage4_lowrank_finetune.py` | SVD low-rank factorization (90% energy, layer3/4) + KD finetune |
| 5 | `training/stage5_quantize.py` | INT8 PTQ (PR/width-informed granularity, CPU fbgemm) |
| 6 | `training/stage6_cv2024_test_eval.py` | Official CV2024 test-set eval (FP32 + INT8): accuracy, macro-F1, Mean AUC, Balanced Accuracy |

Each stage's raw output JSON is in `results/`, checkpoints in `checkpoints/`. See `REPORT.md`
for the full numeric results, per-stage compression trajectory, error analysis, the literature
comparison table (vs. Table II/III), and Modal spend.

## 4. Setup / reproduction

```bash
pip install -r requirements.txt   # only the Modal client + optional local-inspection deps
modal token new                    # one-time Modal auth (see modal.com)
modal secret create huggingface-secret HF_TOKEN=<your HF token>   # for DINOv3 weights

# Run stages in order (each is spawn+detach; poll via `modal app logs <app-id>`
# or `modal volume ls vce-dataset <artifact_dir>` for completion, then download
# with `modal volume get`):
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage0a_download_prep_data.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage0_train_teacher.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage1_kd_alpha075.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage2_prune_surgery.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage3_finetune_pruned.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage4_lowrank_finetune.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage5_quantize.py
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach training/stage6_cv2024_test_eval.py
```

No hardcoded local paths or usernames — everything reads/writes under a fresh Modal volume
namespace (`vce-dataset` volume, `/data/exp015_cv2024_indomain_pipeline/` prefix) and the repo's
own relative `checkpoints/`/`results/` directories. `requirements.txt` lists only what your local
machine needs (the Modal client); heavy deps (torch, transformers, etc.) are declared per-script
and installed automatically inside Modal's remote containers.

## 5. Headline result

Final compressed model (pruned + low-rank + INT8), evaluated on CV2024's official 4,385-image
test set: **Mean AUC 0.723, Balanced Accuracy 0.252, Combined Metric 0.487** — this would place
**~13th of 27** ranked teams on Table II's own combined-metric ranking, and **beats all 6**
Table III organizer baselines (best baseline: ResNet50V2 at 0.360). Model size 0.23 MB, CPU
latency 3.4 ms/image. See `REPORT.md` for the full comparison table and per-class error analysis.
