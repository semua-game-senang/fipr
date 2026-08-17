"""exp015 (in-domain restart) stage 6: final evaluation of the FIPR critical
path's end product (FP32 pre-quant AND INT8 post-quant, from
stage5_quantize.py) on Capsule Vision 2024's OFFICIAL RELEASED TEST SET
(4,385 images, class-labeled folders already present on the volume at
/data/cv2024_test_exp015/ from data acquisition during the first pass --
that's just data, reused here; nothing about the discarded Kvasir-trained
zero-shot RESULTS is reused).

Computes accuracy, macro-F1, AND the challenge's own primary metrics --
Mean AUC (macro one-vs-rest ROC-AUC across the 10 classes) and Balanced
Accuracy (macro-averaged recall) -- exactly as arXiv:2408.04940 Table II/III
report them, for direct apples-to-apples comparison. Also reports full
per-class breakdown (precision/recall/F1/AUC per class), per the project's
standing error-analysis requirement.

Namespace: /data/exp015_cv2024_indomain_pipeline/stage6/

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage6_cv2024_test_eval.py
"""
import json

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
PIPELINE_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
TEST_DIR = f"{DATA_MOUNT}/cv2024_test_exp015/Test set with seperated folders of each class label"
FP32_STATE_DICT = f"{PIPELINE_DIR}/stage5/noquant_fp32_state_dict.pt"
INT8_STATE_DICT = f"{PIPELINE_DIR}/stage5/prquant_int8_state_dict.pt"
STAGE4_CHECKPOINT = f"{PIPELINE_DIR}/stage4/resnet18_alpha075_pruned_lowrank_finetuned_student.pt"
STAGE5_RESULT = f"{PIPELINE_DIR}/stage5/stage5_result.json"
ARTIFACT_DIR = f"{PIPELINE_DIR}/stage6"
RESOLUTION = 224
STEM_WIDTH = 24

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "scikit-learn", "pandas", "numpy", "pillow")
)
app = modal.App("vce-exp015-stage6-cv2024-test-eval", image=image)


@app.function(cpu=4, memory=8192, volumes={DATA_MOUNT: volume}, timeout=3600)
def run():
    import os

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, roc_auc_score,
    )
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

    if not os.path.exists(STAGE4_CHECKPOINT):
        raise RuntimeError(f"stage4 checkpoint not found at {STAGE4_CHECKPOINT}")
    stage4_ckpt = torch.load(STAGE4_CHECKPOINT, map_location=device, weights_only=False)
    classes = stage4_ckpt["classes"]
    stage_widths = stage4_ckpt["stage_widths"]
    target_ranks = stage4_ckpt["target_ranks"]
    print(f"classes ({len(classes)}): {classes}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    with open(STAGE5_RESULT) as f:
        stage5_result = json.load(f)
    per_tensor_prefixes = stage5_result["per_tensor_module_prefixes"]

    # ---------------- test manifest: walk the class-labeled test folders ----------------
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    rows = []
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            raise RuntimeError(f"expected test class folder not found: {cls_dir}")
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith(IMG_EXTS):
                rows.append({"class": cls, "relpath": os.path.join(cls, fn)})
    test_manifest = pd.DataFrame(rows)
    print(f"test set: {len(test_manifest)} images across {test_manifest['class'].nunique()} classes")
    print(test_manifest["class"].value_counts().to_dict())

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    eval_tf = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class CV2024TestDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(TEST_DIR, row["relpath"])
            img = Image.open(path).convert("RGB")
            return eval_tf(img), class_to_idx[row["class"]]

    test_loader = DataLoader(CV2024TestDataset(test_manifest), batch_size=32, shuffle=False, num_workers=4)

    def build_fp32_model():
        m = CompressedResNet(STEM_WIDTH, stage_widths, num_classes=len(classes), target_ranks=target_ranks)
        sd = torch.load(FP32_STATE_DICT, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        return m

    def build_int8_model():
        # Re-run the identical PTQ prepare/convert pipeline from stage5 to
        # reconstruct the quantized graph module, then load its calibrated
        # int8 state dict (FX-quantized modules aren't directly
        # re-instantiable from a bare nn.Module constructor).
        from torch.ao.quantization import QConfigMapping, get_default_qconfig
        from torch.ao.quantization.qconfig import QConfig, default_weight_observer
        from torch.ao.quantization.observer import HistogramObserver
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        fp32_for_quant = CompressedResNet(STEM_WIDTH, stage_widths, num_classes=len(classes), target_ranks=target_ranks)
        fp32_for_quant.load_state_dict(torch.load(FP32_STATE_DICT, map_location=device))
        fp32_for_quant.eval()

        per_tensor_qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=True),
            weight=default_weight_observer,
        )
        qcm = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
        for prefix in per_tensor_prefixes:
            qcm = qcm.set_module_name(prefix, per_tensor_qconfig)

        example_inputs = (torch.randn(1, 3, RESOLUTION, RESOLUTION),)
        prepared = prepare_fx(fp32_for_quant, qcm, example_inputs)
        # Re-calibrate with a handful of forward passes (BN/observer stats
        # were already folded into stage5's saved int8 state dict, but FX
        # requires prepare->convert with SOME calibration pass to build the
        # graph structure before we load the real calibrated weights).
        calib_batch = torch.randn(8, 3, RESOLUTION, RESOLUTION)
        with torch.no_grad():
            prepared(calib_batch)
        converted = convert_fx(prepared)
        converted.load_state_dict(torch.load(INT8_STATE_DICT, map_location=device))
        converted.eval()
        return converted

    def evaluate(model, model_name):
        model.eval()
        all_probs, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.numpy())
                targets.extend(y.numpy().tolist())
        probs = np.concatenate(all_probs, axis=0)
        targets = np.array(targets)
        preds = probs.argmax(axis=1)

        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(targets, preds)

        # Mean AUC: macro one-vs-rest ROC-AUC across all 10 classes (matches
        # challenge protocol). Classes absent from the test set for one-hot
        # binarization would break roc_auc_score's multiclass path -- guard
        # by only including classes present in `targets`.
        present_classes = sorted(set(targets.tolist()))
        y_true_bin = np.zeros((len(targets), len(classes)))
        for i, t in enumerate(targets):
            y_true_bin[i, t] = 1
        per_class_auc = {}
        aucs = []
        for c in range(len(classes)):
            if c not in present_classes or y_true_bin[:, c].sum() == 0 or y_true_bin[:, c].sum() == len(targets):
                per_class_auc[classes[c]] = None
                continue
            try:
                auc_c = roc_auc_score(y_true_bin[:, c], probs[:, c])
            except ValueError:
                auc_c = None
            per_class_auc[classes[c]] = auc_c
            if auc_c is not None:
                aucs.append(auc_c)
        mean_auc = float(np.mean(aucs)) if aucs else None

        report = classification_report(targets, preds, target_names=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))).tolist()

        print(f"[{model_name}] acc={acc:.4f} macro_f1={macro_f1:.4f} balanced_acc={balanced_acc:.4f} mean_auc={mean_auc}")
        return {
            "accuracy": float(acc), "macro_f1": float(macro_f1), "balanced_accuracy": float(balanced_acc),
            "mean_auc": mean_auc, "per_class_auc": per_class_auc,
            "per_class_report": report, "confusion_matrix": cm, "class_order": classes,
            "n_test": len(targets),
        }

    print("Evaluating FP32 (pre-quant) on official CV2024 test set...")
    fp32_model = build_fp32_model()
    fp32_result = evaluate(fp32_model, "FP32")

    print("Evaluating INT8 (post-quant) on official CV2024 test set...")
    int8_model = build_int8_model()
    int8_result = evaluate(int8_model, "INT8")

    result = {
        "test_set": "CV2024 official released test set (4385 images)",
        "n_test": len(test_manifest),
        "class_distribution": test_manifest["class"].value_counts().to_dict(),
        "fp32": fp32_result,
        "int8": int8_result,
    }
    with open(f"{ARTIFACT_DIR}/stage6_result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved: {ARTIFACT_DIR}/stage6_result.json (stage6 COMPLETE)")

    return result


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned call_id={call.object_id}")
