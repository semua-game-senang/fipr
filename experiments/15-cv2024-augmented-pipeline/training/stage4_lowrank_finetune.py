"""exp015 (in-domain restart) stage 4: 90%-Frobenius-energy SVD low-rank
factorization of layer3/layer4 (8 conv layers), applied to THIS experiment's
stage3 pruned+finetuned CV2024 in-domain checkpoint, then fine-tuned on
CV2024's own train/val split, distilling from this experiment's own CV2024
in-domain teacher. Adapted from
experiments/12-alpha075-naive-pruned-lowrank-quant/training/stage3_finetune_lowrank.py
-- same methodology (rank selection: fixed a priori at 90% Frobenius energy,
computed fresh at runtime on the actual stage3 checkpoint's conv weights).

Changes: CV2024 data via manifests, simple in-domain augmentation (matches
stage0/1/3), 8 epochs / batch 64 (budget-matched), CV2024 in-domain
teacher/stage3 checkpoints.

Namespace: /data/exp015_cv2024_indomain_pipeline/stage4/

Usage (spawn-based, poll separately):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage4_lowrank_finetune.py
"""
import json
import os

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
TEACHER_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"
PIPELINE_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
RAW_DIR = f"{PIPELINE_DIR}/raw"
TRAIN_DIR = f"{RAW_DIR}/Dataset/training"
VAL_DIR = f"{RAW_DIR}/Dataset/validation"
TEACHER_CHECKPOINT = f"{PIPELINE_DIR}/stage0/dinov3_teacher_cv2024_indomain.pt"
PRUNED_FINETUNED_CHECKPOINT = f"{PIPELINE_DIR}/stage3/resnet18_alpha075_pruned_r02_finetuned_student.pt"
ARTIFACT_DIR = f"{PIPELINE_DIR}/stage4"

RESOLUTION = 224
N_EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.6
ENERGY_TARGET = 0.90
STEM_WIDTH = 24

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "scikit-learn", "pandas", "numpy", "pillow", "thop")
)
app = modal.App("vce-exp015-stage4-lowrank-finetune-indomain", image=image)


@app.function(gpu="T4", volumes={DATA_MOUNT: volume}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=18000)
def run_finetune():
    import time

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from thop import profile
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, AutoModel

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    label = "cv2024_indomain_alpha075_pruned_r02_lowrank_r90"
    student_checkpoint = f"{ARTIFACT_DIR}/resnet18_alpha075_pruned_lowrank_finetuned_student.pt"

    hf_token = os.environ["HF_TOKEN"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- architecture ----------------
    class BasicBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class FactoredConv(nn.Module):
        def __init__(self, in_ch, out_ch, r, kernel_size, stride, padding):
            super().__init__()
            self.r = r
            self.conv_V = nn.Conv2d(in_ch, r, kernel_size, stride=stride, padding=padding, bias=False)
            self.conv_U = nn.Conv2d(r, out_ch, 1, stride=1, bias=False)

        def forward(self, x):
            return self.conv_U(self.conv_V(x))

    class FactoredBasicBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride, r1, r2):
            super().__init__()
            self.conv1 = FactoredConv(in_ch, out_ch, r1, 3, stride, 1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = FactoredConv(out_ch, out_ch, r2, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class NarrowResNetPlain(nn.Module):
        def __init__(self, stem_width, stage_widths, num_classes, blocks_per_stage=(2, 2, 2, 2)):
            super().__init__()
            self.conv1 = nn.Conv2d(3, stem_width, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_width)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            in_ch = stem_width
            stages = []
            for i, (out_ch, n_blocks) in enumerate(zip(stage_widths, blocks_per_stage)):
                stride = 1 if i == 0 else 2
                blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
                for _ in range(n_blocks - 1):
                    blocks.append(BasicBlock(out_ch, out_ch, stride=1))
                stages.append(nn.Sequential(*blocks))
                in_ch = out_ch
            self.layer1, self.layer2, self.layer3, self.layer4 = stages
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(stage_widths[-1], num_classes)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = torch.flatten(self.avgpool(x), 1)
            return self.fc(x)

    class NarrowResNetLowRank(nn.Module):
        def __init__(self, stem_width, stage_widths, num_classes, target_ranks: dict):
            super().__init__()
            w0, w1, w2, w3 = stage_widths
            self.conv1 = nn.Conv2d(3, stem_width, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_width)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = nn.Sequential(BasicBlock(stem_width, w0, stride=1), BasicBlock(w0, w0, stride=1))
            self.layer2 = nn.Sequential(BasicBlock(w0, w1, stride=2), BasicBlock(w1, w1, stride=1))
            self.layer3 = nn.Sequential(
                FactoredBasicBlock(w1, w2, stride=2,
                                    r1=target_ranks["layer3.0.conv1"], r2=target_ranks["layer3.0.conv2"]),
                FactoredBasicBlock(w2, w2, stride=1,
                                    r1=target_ranks["layer3.1.conv1"], r2=target_ranks["layer3.1.conv2"]),
            )
            self.layer4 = nn.Sequential(
                FactoredBasicBlock(w2, w3, stride=2,
                                    r1=target_ranks["layer4.0.conv1"], r2=target_ranks["layer4.0.conv2"]),
                FactoredBasicBlock(w3, w3, stride=1,
                                    r1=target_ranks["layer4.1.conv1"], r2=target_ranks["layer4.1.conv2"]),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(w3, num_classes)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = torch.flatten(self.avgpool(x), 1)
            return self.fc(x)

    # ---------------- load stage3 finetuned-pruned checkpoint ----------------
    if not os.path.exists(PRUNED_FINETUNED_CHECKPOINT):
        raise RuntimeError(f"Pruned+finetuned checkpoint not found at {PRUNED_FINETUNED_CHECKPOINT}")
    pruned_ckpt = torch.load(PRUNED_FINETUNED_CHECKPOINT, map_location=device, weights_only=False)
    classes = pruned_ckpt["classes"]
    stage_widths = pruned_ckpt["stage_widths"]
    print(f"loaded stage3 finetuned-pruned checkpoint: stage_widths={stage_widths} "
          f"acc={pruned_ckpt.get('best_val_accuracy')} macro_f1={pruned_ckpt.get('best_val_macro_f1')}")

    original = NarrowResNetPlain(STEM_WIDTH, stage_widths, num_classes=len(classes)).to(device)
    original.load_state_dict(pruned_ckpt["state_dict"])
    original.eval()

    def r90_rank(W: torch.Tensor, target_energy: float = ENERGY_TARGET) -> int:
        out_ch, in_ch, kh, kw = W.shape
        W2d = W.reshape(out_ch, in_ch * kh * kw).double()
        S = torch.linalg.svdvals(W2d)
        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        r = int((energy >= target_energy).nonzero()[0].item()) + 1
        return max(1, min(r, min(out_ch, in_ch * kh * kw)))

    target_ranks = {}
    for stage in ("layer3", "layer4"):
        for idx in (0, 1):
            block = getattr(original, stage)[idx]
            target_ranks[f"{stage}.{idx}.conv1"] = r90_rank(block.conv1.weight.data)
            target_ranks[f"{stage}.{idx}.conv2"] = r90_rank(block.conv2.weight.data)
    print(f"fresh r90 ranks: {target_ranks}")

    student = NarrowResNetLowRank(STEM_WIDTH, stage_widths, num_classes=len(classes), target_ranks=target_ranks).to(device)

    unchanged_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "fc.")
    orig_sd = original.state_dict()
    student_sd = student.state_dict()
    copied = 0
    for k in list(student_sd.keys()):
        if k.startswith(unchanged_prefixes) and k in orig_sd and student_sd[k].shape == orig_sd[k].shape:
            student_sd[k] = orig_sd[k].clone()
            copied += 1
    student.load_state_dict(student_sd)
    print(f"Copied {copied} unchanged tensors (stem/layer1/layer2/fc)")

    def get_block(model, stage, idx):
        return getattr(model, stage)[idx]

    def warm_start_factored_conv(orig_conv, factored):
        W = orig_conv.weight.detach().clone()
        out_ch, in_ch, kh, kw = W.shape
        r = factored.r
        W2d = W.reshape(out_ch, in_ch * kh * kw).double()
        U, S, Vt = torch.linalg.svd(W2d, full_matrices=False)
        max_rank = min(out_ch, in_ch * kh * kw)
        r_eff = min(r, max_rank)
        Vr = Vt[:r_eff, :].reshape(r_eff, in_ch, kh, kw).to(W.dtype)
        Ur = (U[:, :r_eff] * S[:r_eff]).reshape(out_ch, r_eff, 1, 1).to(W.dtype)
        with torch.no_grad():
            factored.conv_V.weight.copy_(Vr)
            factored.conv_U.weight.copy_(Ur)
        rel_err = float(
            torch.linalg.norm((U[:, :r_eff] @ torch.diag(S[:r_eff]) @ Vt[:r_eff, :]) - W2d) / torch.linalg.norm(W2d)
        )
        return rel_err

    warm_start_diagnostics = {}
    for stage in ("layer3", "layer4"):
        for idx in (0, 1):
            orig_block = get_block(original, stage, idx)
            student_block = get_block(student, stage, idx)
            student_block.bn1.load_state_dict(orig_block.bn1.state_dict())
            student_block.bn2.load_state_dict(orig_block.bn2.state_dict())
            if orig_block.downsample is not None:
                student_block.downsample.load_state_dict(orig_block.downsample.state_dict())
            rel_err1 = warm_start_factored_conv(orig_block.conv1, student_block.conv1)
            rel_err2 = warm_start_factored_conv(orig_block.conv2, student_block.conv2)
            warm_start_diagnostics[f"{stage}.{idx}.conv1"] = rel_err1
            warm_start_diagnostics[f"{stage}.{idx}.conv2"] = rel_err2
            print(f"warm-started {stage}.{idx}: conv1 rel_err={rel_err1:.4f} conv2 rel_err={rel_err2:.4f}")

    student.eval()
    verify_batch = torch.randn(4, 3, RESOLUTION, RESOLUTION, device=device)
    with torch.no_grad():
        full_out_orig = original(verify_batch)
        full_out_stu = student(verify_batch)
        full_output_rel_err = float(torch.linalg.norm(full_out_stu - full_out_orig) / torch.linalg.norm(full_out_orig))
    print(f"Numerical verification: full_model_logits_rel_err={full_output_rel_err:.4f}")

    # ---------------- data ----------------
    train_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_train.csv")
    val_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_val.csv")
    assert classes == sorted(train_manifest["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if not os.path.exists(TEACHER_CHECKPOINT):
        raise RuntimeError(f"Teacher checkpoint not found at {TEACHER_CHECKPOINT}")
    teacher_ckpt = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
    teacher_processor = AutoImageProcessor.from_pretrained(TEACHER_BACKBONE, token=hf_token)
    teacher_backbone = AutoModel.from_pretrained(TEACHER_BACKBONE, token=hf_token).to(device)

    class TeacherModel(nn.Module):
        def __init__(self, backbone, hidden_size, num_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values)
            return self.classifier(out.last_hidden_state[:, 0, :])

    teacher = TeacherModel(teacher_backbone, teacher_ckpt["hidden_size"], len(classes)).to(device)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---------------- simple in-domain augmentation (matches stage0/1/3) ----------------
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    student_train_tf = T.Compose([
        T.RandomResizedCrop(RESOLUTION, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    student_eval_tf = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class CV2024Dataset(Dataset):
        def __init__(self, df, root_dir):
            self.df = df.reset_index(drop=True)
            self.root_dir = root_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(self.root_dir, row["relpath"])
            img = Image.open(path).convert("RGB")
            return img, class_to_idx[row["class"]]

    def make_collate(train: bool):
        student_tf = student_train_tf if train else student_eval_tf

        def collate(batch):
            imgs, labels = zip(*batch)
            student_pv = torch.stack([student_tf(im) for im in imgs])
            teacher_pv = teacher_processor(images=list(imgs), return_tensors="pt")["pixel_values"]
            return student_pv, teacher_pv, torch.tensor(labels)
        return collate

    train_loader = DataLoader(CV2024Dataset(train_manifest, TRAIN_DIR), batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=6, pin_memory=True, drop_last=True, collate_fn=make_collate(True))
    val_loader = DataLoader(CV2024Dataset(val_manifest, VAL_DIR), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=6, collate_fn=make_collate(False))

    class_counts = train_manifest["class"].value_counts()
    freqs = np.array([class_counts[c] for c in classes], dtype=np.float64)
    weights = 1.0 / np.sqrt(freqs)
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    def eval_student():
        student.eval()
        preds, targets = [], []
        with torch.no_grad():
            for student_pv, _, labels in val_loader:
                logits = student(student_pv.to(device))
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                targets.extend(labels.numpy().tolist())
        preds, targets = np.array(preds), np.array(targets)
        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        return float(acc), float(macro_f1)

    pre_ft_acc, pre_ft_macro_f1 = eval_student()
    print(f"Pre-fine-tune (warm-started factorized, 0 fine-tune steps): acc={pre_ft_acc:.4f} macro_f1={pre_ft_macro_f1:.4f}")

    # ---------------- fine-tune ----------------
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    ce_criterion = nn.CrossEntropyLoss(weight=weights_t)

    per_epoch_summary = []
    best_macro_f1 = -1.0
    best_state = best_report = best_cm = None
    best_epoch = -1
    best_acc = None

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        student.train()
        t0 = time.time()
        running_kd, running_ce, running_total = 0.0, 0.0, 0.0
        for step, (student_pv, teacher_pv, labels) in enumerate(train_loader):
            student_pv, teacher_pv, labels = student_pv.to(device), teacher_pv.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(teacher_pv)
            student_logits = student(student_pv)
            kd_loss = Fnn.kl_div(
                Fnn.log_softmax(student_logits / KD_TEMPERATURE, dim=1),
                Fnn.softmax(teacher_logits / KD_TEMPERATURE, dim=1),
                reduction="batchmean",
            ) * (KD_TEMPERATURE ** 2)
            ce_loss = ce_criterion(student_logits, labels)
            total_loss = KD_ALPHA * kd_loss + (1 - KD_ALPHA) * ce_loss
            total_loss.backward()
            optimizer.step()
            bs = student_pv.size(0)
            running_kd += kd_loss.item() * bs
            running_ce += ce_loss.item() * bs
            running_total += total_loss.item() * bs
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} total_loss={total_loss.item():.4f}")
        scheduler.step()
        n_train = len(train_manifest)
        train_kd, train_ce, train_total = running_kd / n_train, running_ce / n_train, running_total / n_train

        acc, macro_f1 = eval_student()
        preds_for_report_acc, preds_for_report_f1 = acc, macro_f1
        epoch_time = time.time() - t0

        # per-class report/cm recomputed only when this epoch is a new best (below)
        per_epoch_summary.append({
            "epoch": epoch, "train_kd_loss": train_kd, "train_ce_loss": train_ce, "train_total_loss": train_total,
            "val_accuracy": acc, "val_macro_f1": macro_f1,
            "lr": scheduler.get_last_lr()[0], "epoch_seconds": epoch_time,
        })
        print(f"[{label}] epoch {epoch}: total={train_total:.4f} acc={acc:.4f} macro_f1={macro_f1:.4f} ({epoch_time:.1f}s)")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            student.eval()
            preds, targets = [], []
            with torch.no_grad():
                for student_pv, _, labels in val_loader:
                    logits = student(student_pv.to(device))
                    preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    targets.extend(labels.numpy().tolist())
            preds, targets = np.array(preds), np.array(targets)
            best_report = classification_report(targets, preds, target_names=classes, output_dict=True, zero_division=0)
            best_cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))).tolist()
            best_acc, best_epoch = acc, epoch

        try:
            torch.save({"state_dict": best_state, "classes": classes, "best_epoch": best_epoch,
                        "best_val_accuracy": best_acc, "best_val_macro_f1": best_macro_f1,
                        "label": label, "stem_width": STEM_WIDTH, "stage_widths": stage_widths,
                        "target_ranks": target_ranks, "training_complete": False}, student_checkpoint)
            volume.commit()
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: mid-training checkpoint commit failed at epoch {epoch} ({e}); continuing")

    total_wall_seconds = time.time() - t_start
    student.load_state_dict(best_state)
    student.eval()

    torch.save({"state_dict": best_state, "classes": classes, "best_epoch": best_epoch,
                "best_val_accuracy": best_acc, "best_val_macro_f1": best_macro_f1,
                "label": label, "stem_width": STEM_WIDTH, "stage_widths": stage_widths,
                "target_ranks": target_ranks, "training_complete": True}, student_checkpoint)
    volume.commit()

    dummy = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(device)
    macs, params = profile(student, inputs=(dummy,), verbose=False)
    flops_fwd = 2 * macs
    fp32_size_mb = params * 4 / (1024 ** 2)
    flops_row = {"model": f"resnet18_{label}_finetuned_from_dinov3s", "label": label,
                 "stem_width": STEM_WIDTH, "stage_widths": stage_widths, "target_ranks": target_ranks,
                 "params_M": params / 1e6, "fp32_size_mb": fp32_size_mb, "flops_G_per_image_fwd": flops_fwd / 1e9,
                 "gpu": "T4"}
    print(f"flops_audit: params_M={params/1e6:.3f} flops_G_fwd={flops_fwd/1e9:.4f}")

    result = {
        "model": f"resnet18_{label}_finetuned_from_dinov3s", "label": label,
        "stem_width": STEM_WIDTH, "stage_widths": stage_widths, "target_ranks": target_ranks,
        "n_epochs": N_EPOCHS, "n_train": len(train_manifest), "n_val": len(val_manifest),
        "warm_start_svd_relative_frobenius_error": warm_start_diagnostics,
        "numerical_verification": {"full_model_logits_rel_err_vs_pruned_source": full_output_rel_err},
        "pre_finetune_warmstarted_eval": {"accuracy": pre_ft_acc, "macro_f1": pre_ft_macro_f1},
        "per_epoch_summary": per_epoch_summary,
        "best_epoch": best_epoch, "best_val_accuracy": best_acc, "best_val_macro_f1": best_macro_f1,
        "per_class_report": best_report, "confusion_matrix": best_cm, "class_order": classes,
        "flops_audit": flops_row,
        "student_checkpoint_path": student_checkpoint, "total_wall_seconds": total_wall_seconds,
        "source_checkpoint_best_val_accuracy": pruned_ckpt.get("best_val_accuracy"),
        "source_checkpoint_best_val_macro_f1": pruned_ckpt.get("best_val_macro_f1"),
    }

    with open(f"{ARTIFACT_DIR}/stage4_result.json", "w") as f:
        json.dump(result, f)
    volume.commit()
    print(f"Saved: {ARTIFACT_DIR}/stage4_result.json (stage4 COMPLETE) "
          f"best_val_accuracy={best_acc:.4f} macro_f1={best_macro_f1:.4f}")

    return result


@app.local_entrypoint()
def main():
    call = run_finetune.spawn()
    print(f"Spawned run_finetune as FunctionCall id={call.object_id}")
    print(f"Poll for completion via: modal volume ls vce-dataset {ARTIFACT_DIR.lstrip('/')}")
