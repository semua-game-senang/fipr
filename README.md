# fipr

Packaged deliverables for the FIPR (fine-tune → interpolated-width distillation → prune →
low-rank) compression pipeline, trained and evaluated in-domain on the Capsule Vision 2024
Challenge's own train/validation/test split (arXiv:2408.04940).

This is a standalone extract of `experiments/15-cv2024-augmented-pipeline/` from the parent
`vce` research repository, packaged separately for public release. See
`experiments/15-cv2024-augmented-pipeline/README.md` for the full pipeline description and
setup/reproduction instructions, and `experiments/15-cv2024-augmented-pipeline/REPORT.md` for
the complete results, error analysis, and literature comparison against the challenge's Table II
(27 ranked teams) and Table III (6 organizer baselines).

## Contents

- `experiments/15-cv2024-augmented-pipeline/training/` — all 8 pipeline stage scripts (Modal-based)
- `experiments/15-cv2024-augmented-pipeline/checkpoints/` — compressed-pipeline checkpoints (stage 1 onward; the ~86 MB teacher checkpoint is excluded, see the experiment README)
- `experiments/15-cv2024-augmented-pipeline/results/` — raw per-stage result JSON (metrics, per-class breakdowns, confusion matrices)
- `experiments/15-cv2024-augmented-pipeline/README.md` / `REPORT.md` — full writeup

## Headline result

Final compressed model (pruned + low-rank + INT8): **0.189M params, 0.233 MB, 3.4 ms/image
(CPU)**, evaluated on the official 4,385-image CV2024 test set at **Mean AUC 0.723 / Balanced
Accuracy 0.252** — ranking ~13th of 27 challenge teams on the challenge's own combined metric,
and beating all 6 organizer CNN baselines.
