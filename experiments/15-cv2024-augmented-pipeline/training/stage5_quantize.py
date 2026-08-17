"""exp015 (in-domain restart) stage 5: PR/width-ranked INT8 PTQ of the
"alpha=0.75 + pruned + low-rank, no quant" CV2024 in-domain checkpoint from
stage4_lowrank_finetune.py. Adapted from
experiments/12-alpha075-naive-pruned-lowrank-quant/training/stage5_quantize.py
-- same methodology (fresh per-layer PR/width profile on THIS checkpoint's
own activations, 3-of-5 layer1/2/3/4/fc groups ranked ascending by PR/width
-> lowest 3 demoted to per-tensor INT8, highest 2 stay per-channel).
Calibration: 400 images from CV2024's own TRAIN split, seed=0. CPU (fbgemm).

Sanity-check evaluation here uses CV2024's own VALIDATION split (matching
the established convention of this codebase); the official CV2024 TEST-set
evaluation (Mean AUC / Balanced Accuracy, matching the challenge's own
protocol) is a separate stage (stage6_cv2024_test_eval.py).

Namespace: /data/exp015_cv2024_indomain_pipeline/stage5/

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage5_quantize.py
"""
import json

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
PIPELINE_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
RAW_DIR = f"{PIPELINE_DIR}/raw"
TRAIN_DIR = f"{RAW_DIR}/Dataset/training"
VAL_DIR = f"{RAW_DIR}/Dataset/validation"
ARTIFACT_DIR = f"{PIPELINE_DIR}/stage5"
RESOLUTION = 224
CALIB_SAMPLE_SIZE = 400
PR_PROFILE_SAMPLES_PER_CLASS = 50
LATENCY_RUNS = 50
LATENCY_WARMUP = 10
STEM_WIDTH = 24
FP32_CHECKPOINT = f"{PIPELINE_DIR}/stage4/resnet18_alpha075_pruned_lowrank_finetuned_student.pt"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "scikit-learn", "pandas", "numpy", "pillow")
)
app = modal.App("vce-exp015-stage5-quantize-indomain", image=image)


@app.function(cpu=4, memory=8192, volumes={DATA_MOUNT: volume}, timeout=3600)
def run():
    import io
    import os
    import time

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch.ao.quantization import QConfigMapping, get_default_qconfig
    from torch.ao.quantization.qconfig import QConfig, default_weight_observer
    from torch.ao.quantization.observer import HistogramObserver
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torch.utils.data import DataLoader, Dataset

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.backends.quantized.engine = "fbgemm"
    device = "cpu"

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

    class CompressedResNet(nn.Module):
        def __init__(self, stem_width, stage_widths, num_classes, target_ranks):
            super().__init__()
            self.conv1 = nn.Conv2d(3, stem_width, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_width)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            w0, w1, w2, w3 = stage_widths
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

    if not os.path.exists(FP32_CHECKPOINT):
        raise RuntimeError(f"checkpoint not found at {FP32_CHECKPOINT}")
    ckpt = torch.load(FP32_CHECKPOINT, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    stage_widths = ckpt["stage_widths"]
    target_ranks = ckpt["target_ranks"]
    print(f"loaded stage4 checkpoint: stage_widths={stage_widths} acc={ckpt.get('best_val_accuracy'):.4f} "
          f"macro_f1={ckpt.get('best_val_macro_f1'):.4f}")

    train_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_train.csv")
    val_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_val.csv")
    assert classes == sorted(train_manifest["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    eval_tf = T.Compose([
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
            return eval_tf(img), class_to_idx[row["class"]], row["relpath"]

    def collate(batch):
        xs, ys, keys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys), list(keys)

    # CPU-only sanity-check eval: subsample val (stratified) to keep this
    # stage fast -- full official numbers on the CV2024 TEST set (all
    # images) come from stage6_cv2024_test_eval.py, which is the
    # authoritative evaluation for the paper's comparison table.
    VAL_SANITY_SAMPLES_PER_CLASS = 200
    val_sanity_manifest = pd.concat([
        g.sample(min(len(g), VAL_SANITY_SAMPLES_PER_CLASS), random_state=0)
        for _, g in val_manifest.groupby("class")
    ]).reset_index(drop=True)
    print(f"CPU sanity-check val subsample: {len(val_sanity_manifest)} images (of {len(val_manifest)} full val)")
    val_loader = DataLoader(CV2024Dataset(val_sanity_manifest, VAL_DIR), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)

    def build_fp32_model():
        m = CompressedResNet(STEM_WIDTH, stage_widths, num_classes=len(classes), target_ranks=target_ranks)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m

    # --- fresh PR/width profile on THIS checkpoint (CV2024 val images) ---
    pr_sample = pd.concat([g.sample(min(len(g), PR_PROFILE_SAMPLES_PER_CLASS), random_state=0)
                            for _, g in val_manifest.groupby("class")])
    pr_images = []
    for _, r in pr_sample.iterrows():
        path = os.path.join(VAL_DIR, r["relpath"])
        pr_images.append(eval_tf(Image.open(path).convert("RGB")))
    print(f"PR-profile sample: {len(pr_images)} images")

    pr_model = build_fp32_model()
    activations = {}

    def make_hook(hook_name, spatial):
        def hook(module, inp, output):
            feat = output.mean(dim=[2, 3]) if spatial else output.flatten(1)
            activations.setdefault(hook_name, []).append(feat.detach().numpy())
        return hook

    hook_handles = [
        pr_model.layer1.register_forward_hook(make_hook("layer1", True)),
        pr_model.layer2.register_forward_hook(make_hook("layer2", True)),
        pr_model.layer3.register_forward_hook(make_hook("layer3", True)),
        pr_model.layer4.register_forward_hook(make_hook("layer4", True)),
        pr_model.avgpool.register_forward_hook(make_hook("fc", False)),
    ]
    with torch.no_grad():
        for start in range(0, len(pr_images), 64):
            batch = torch.stack(pr_images[start:start + 64])
            pr_model(batch)
    for h in hook_handles:
        h.remove()

    def pr_over_width(X: np.ndarray) -> float:
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
        eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
        pr = float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))
        return pr / X.shape[1]

    pr_profile = {}
    for group in ["layer1", "layer2", "layer3", "layer4", "fc"]:
        X = np.concatenate(activations[group], axis=0)
        pr_profile[group] = pr_over_width(X)
    print(f"fresh PR/width profile: {pr_profile}")

    ranked = sorted(pr_profile.items(), key=lambda kv: kv[1])
    per_tensor_prefixes = [g for g, _ in ranked[:3]]
    per_channel_prefixes = [g for g, _ in ranked[3:]]
    print(f"per-tensor (most redundant): {per_tensor_prefixes} | per-channel (least redundant): {per_channel_prefixes}")

    # --- calibration + quantization (CV2024 train split) ---
    rng = np.random.RandomState(0)
    calib_idx = rng.choice(len(train_manifest), size=min(CALIB_SAMPLE_SIZE, len(train_manifest)), replace=False)
    calib_images = []
    for i in calib_idx:
        row = train_manifest.iloc[i]
        path = os.path.join(TRAIN_DIR, row["relpath"])
        calib_images.append(eval_tf(Image.open(path).convert("RGB")))
    print(f"calibration set: {len(calib_images)} images")
    example_inputs = (torch.stack(calib_images[:1]),)

    def calibrate(prepared):
        prepared.eval()
        with torch.no_grad():
            for start in range(0, len(calib_images), 32):
                batch = torch.stack(calib_images[start:start + 32])
                prepared(batch)
        return prepared

    def build_step3_qconfig_mapping():
        per_tensor_qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=True),
            weight=default_weight_observer,
        )
        qcm = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
        for prefix in per_tensor_prefixes:
            qcm = qcm.set_module_name(prefix, per_tensor_qconfig)
        return qcm

    def quantize(qconfig_mapping):
        model = build_fp32_model()
        prepared = prepare_fx(model, qconfig_mapping, example_inputs)
        prepared = calibrate(prepared)
        converted = convert_fx(prepared)
        converted.eval()
        return converted

    def evaluate_full(model):
        preds, targets = [], []
        with torch.no_grad():
            for x, labels, keys in val_loader:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                preds.extend(pred.numpy().tolist())
                targets.extend(labels.numpy().tolist())
        preds, targets = np.array(preds), np.array(targets)
        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        cm = confusion_matrix(targets, preds, labels=range(len(classes))).tolist()
        return {"val_accuracy": float(acc), "val_macro_f1": float(macro_f1),
                "class_order": classes, "confusion_matrix": cm}

    def state_dict_size_mb(model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return len(buf.getvalue()) / (1024 ** 2)

    def measure_latency(model):
        x = torch.stack(calib_images[:1])
        with torch.no_grad():
            for _ in range(LATENCY_WARMUP):
                model(x)
            times = []
            for _ in range(LATENCY_RUNS):
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000)
        return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)), "median_ms": float(np.median(times))}

    print("Evaluating FP32 on CV2024 val (sanity check)...")
    fp32_model = build_fp32_model()
    fp32_metrics = evaluate_full(fp32_model)
    fp32_size_mb = state_dict_size_mb(fp32_model)
    fp32_latency = measure_latency(fp32_model)
    print(f"FP32: acc={fp32_metrics['val_accuracy']:.4f} macro_f1={fp32_metrics['val_macro_f1']:.4f} "
          f"size_mb={fp32_size_mb:.3f} latency={fp32_latency}")

    print("Quantizing + evaluating PR-informed granularity INT8...")
    int8_model = quantize(build_step3_qconfig_mapping())
    int8_metrics = evaluate_full(int8_model)
    int8_size_mb = state_dict_size_mb(int8_model)
    int8_latency = measure_latency(int8_model)
    print(f"INT8: acc={int8_metrics['val_accuracy']:.4f} macro_f1={int8_metrics['val_macro_f1']:.4f} "
          f"size_mb={int8_size_mb:.3f} latency={int8_latency}")

    fp32_ckpt_path = f"{ARTIFACT_DIR}/noquant_fp32_state_dict.pt"
    int8_ckpt_path = f"{ARTIFACT_DIR}/prquant_int8_state_dict.pt"
    torch.save(fp32_model.state_dict(), fp32_ckpt_path)
    torch.save(int8_model.state_dict(), int8_ckpt_path)
    volume.commit()

    result = {
        "checkpoint": FP32_CHECKPOINT, "stage_widths": stage_widths, "target_ranks": target_ranks,
        "n_train": len(train_manifest), "n_val_full": len(val_manifest), "n_val_sanity_subsample": len(val_sanity_manifest),
        "n_calibration_images": len(calib_images), "quantization_engine": "fbgemm",
        "fresh_pr_profile": pr_profile,
        "per_tensor_module_prefixes": per_tensor_prefixes,
        "per_channel_module_prefixes": per_channel_prefixes,
        "fp32": {"metrics": fp32_metrics, "size_mb": fp32_size_mb, "latency_ms": fp32_latency},
        "int8_pr_informed_granularity": {"metrics": int8_metrics, "size_mb": int8_size_mb, "latency_ms": int8_latency},
        "fp32_checkpoint_path": fp32_ckpt_path, "int8_checkpoint_path": int8_ckpt_path,
    }
    with open(f"{ARTIFACT_DIR}/stage5_result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved: {ARTIFACT_DIR}/stage5_result.json (stage5 COMPLETE)")

    return result


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned call_id={call.object_id}")
