"""exp015 (in-domain restart) stage 0: DINOv3-S teacher fine-tune trained
on Capsule Vision 2024's OWN train/validation split (10-class taxonomy),
downloaded/prepared by stage0a_download_prep_data.py. This is a from-scratch
retrain of the FIPR critical path's teacher stage, matching the challenge's
own training protocol exactly (not the discarded Kvasir-zero-shot pass).

Adapted from experiments/02-distillation/exp002-naive-split-replication/
training/train_teacher_naive.py, with:
  1. Data: reads CV2024's own train/ and validation/ folders directly (via
     the manifests stage0a wrote), 10-class head (angioectasia, bleeding,
     erosion, erythema, foreign body, lymphangiectasia, normal, polyp,
     ulcer, worms) instead of the 14-class Kvasir taxonomy. Their official
     train/val boundary is used as-is -- no re-splitting.
  2. Augmentation: simple, standard, non-domain-gap-driven (training is now
     in-domain): horizontal+vertical flip (no canonical VCE orientation),
     mild rotation, brightness/contrast/saturation jitter. No blur/noise --
     this isn't the load-bearing design decision this time.
  3. Class imbalance: CV2024's train set is ~76% "Normal" -- inverse-sqrt-
     frequency class weighting in the loss, same pattern as prior stages.
  4. GPU: T4. Secret: modal.Secret.from_name("huggingface-secret").

Namespace: /data/exp015_cv2024_indomain_pipeline/stage0/
(local checkpoints/results under experiments/15-cv2024-augmented-pipeline/).

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage0_train_teacher.py
"""
import json

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"
RESOLUTION = 224
N_EPOCHS = 6
BATCH_SIZE = 96
HEAD_LR = 1e-3
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
PIPELINE_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
RAW_DIR = f"{PIPELINE_DIR}/raw"
TRAIN_DIR = f"{RAW_DIR}/Dataset/training"
VAL_DIR = f"{RAW_DIR}/Dataset/validation"
ARTIFACT_DIR = f"{PIPELINE_DIR}/stage0"
CHECKPOINT_PATH = f"{ARTIFACT_DIR}/dinov3_teacher_cv2024_indomain.pt"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "scikit-learn", "pandas", "numpy", "pillow", "thop")
)
app = modal.App("vce-exp015-stage0-train-teacher-indomain", image=image)


@app.function(gpu="T4", volumes={DATA_MOUNT: volume}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=18000)
def run():
    import os
    import time

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from transformers import AutoImageProcessor, AutoModel

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    hf_token = os.environ["HF_TOKEN"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_train.csv")
    val_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_val.csv")
    classes = sorted(train_manifest["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"classes ({len(classes)}): {classes}")
    print(f"train={len(train_manifest)} val={len(val_manifest)}")

    processor = AutoImageProcessor.from_pretrained(BACKBONE, token=hf_token)
    backbone = AutoModel.from_pretrained(BACKBONE, token=hf_token).to(device)
    hidden_size = backbone.config.hidden_size

    # ---------------- simple, in-domain-appropriate augmentation ----------------
    # No canonical VCE frame orientation -> flips justified. Mild rotation and
    # photometric jitter for general robustness. Kept deliberately simple:
    # in-domain training is the load-bearing design choice this time, not
    # aggressive augmentation.
    pil_augment_tf = T.Compose([
        T.RandomResizedCrop(RESOLUTION, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    ])

    class CV2024Dataset(Dataset):
        def __init__(self, df, root_dir, train: bool):
            self.df = df.reset_index(drop=True)
            self.root_dir = root_dir
            self.train = train

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # relpath is already relative to root_dir (see stage0a manifest builder)
            path = os.path.join(self.root_dir, row["relpath"])
            img = Image.open(path).convert("RGB")
            if self.train:
                img = pil_augment_tf(img)
            return img, class_to_idx[row["class"]]

    def collate(batch):
        imgs, labels = zip(*batch)
        pixel_values = processor(images=list(imgs), return_tensors="pt")["pixel_values"]
        return pixel_values, torch.tensor(labels)

    train_loader = DataLoader(CV2024Dataset(train_manifest, TRAIN_DIR, True), batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(CV2024Dataset(val_manifest, VAL_DIR, False), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, collate_fn=collate)

    class_counts = train_manifest["class"].value_counts()
    freqs = np.array([class_counts[c] for c in classes], dtype=np.float64)
    weights = 1.0 / np.sqrt(freqs)
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"class freqs: {dict(zip(classes, freqs.tolist()))}")
    print(f"class weights: {dict(zip(classes, weights.tolist()))}")

    class FineTuneModel(nn.Module):
        def __init__(self, backbone, hidden_size, num_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values)
            return self.classifier(out.last_hidden_state[:, 0, :])

    model = FineTuneModel(backbone, hidden_size, len(classes)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.classifier.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=weights_t)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    best_acc = -1.0
    per_epoch_summary = []
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, (pixel_values, labels) in enumerate(train_loader):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * pixel_values.size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                logits = model(pixel_values.to(device))
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                targets.extend(labels.numpy().tolist())
        preds, targets = np.array(preds), np.array(targets)
        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        epoch_time = time.time() - t0
        print(f"epoch {epoch}: loss={running_loss/len(train_manifest):.4f} acc={acc:.4f} macro_f1={macro_f1:.4f} ({epoch_time:.1f}s)")
        per_epoch_summary.append({"epoch": epoch, "train_loss": running_loss / len(train_manifest),
                                   "val_accuracy": float(acc), "val_macro_f1": float(macro_f1),
                                   "epoch_seconds": epoch_time})

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_acc = acc
            best_epoch = epoch
            # Save best-so-far after every epoch -- protects against losing
            # all progress if the run gets interrupted or times out partway
            # through (long T4 run: ~20-25 min/epoch).
            torch.save({
                "state_dict": best_state, "hidden_size": hidden_size, "num_classes": len(classes),
                "classes": classes, "backbone_hf_id": BACKBONE, "best_epoch": best_epoch,
                "best_val_accuracy": best_acc, "best_val_macro_f1": best_macro_f1,
                "training_protocol": "cv2024_indomain_own_train_val_split",
            }, CHECKPOINT_PATH)
            volume.commit()
            print(f"  [checkpoint] saved best-so-far at epoch {epoch} (macro_f1={macro_f1:.4f})")
        with open(f"{ARTIFACT_DIR}/per_epoch_summary_partial.json", "w") as f:
            json.dump(per_epoch_summary, f, indent=2)
        volume.commit()

    total_wall_seconds = time.time() - t_start
    print(f"Best epoch {best_epoch}: acc={best_acc:.4f} macro_f1={best_macro_f1:.4f} total_wall_seconds={total_wall_seconds:.1f}")
    torch.save({
        "state_dict": best_state, "hidden_size": hidden_size, "num_classes": len(classes),
        "classes": classes, "backbone_hf_id": BACKBONE, "best_epoch": best_epoch,
        "best_val_accuracy": best_acc, "best_val_macro_f1": best_macro_f1,
        "training_protocol": "cv2024_indomain_own_train_val_split",
    }, CHECKPOINT_PATH)
    volume.commit()
    print(f"Saved teacher checkpoint to {CHECKPOINT_PATH}")

    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt_bytes = f.read()

    return {"best_epoch": best_epoch, "best_val_accuracy": float(best_acc),
            "best_val_macro_f1": float(best_macro_f1),
            "checkpoint_path": CHECKPOINT_PATH,
            "per_epoch_summary": per_epoch_summary, "total_wall_seconds": total_wall_seconds,
            "gpu": "T4", "n_train": len(train_manifest), "n_val": len(val_manifest),
            "classes": classes,
            "_ckpt_bytes": ckpt_bytes}


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned call_id={call.object_id}")
