"""exp015 (in-domain restart) stage 2: PR+Fisher(GE) combined-score
structured pruning surgery (ratio=0.20) on THIS experiment's own alpha=0.75
CV2024 in-domain KD student (stage1_kd_alpha075.py output), no fine-tuning
yet (that is stage3). Adapted from
experiments/12-alpha075-naive-pruned-lowrank-quant/training/stage1_profile_and_prune.py
-- same methodology (PR-contribution-share + Fisher/GE-share combined
channel score, layer2/layer3/layer4 pruned, layer1+stem left full-width),
applied to the CV2024 in-domain checkpoint. GPU T4, lightweight
profiling+surgery step (no full training loop). Profiling/eval uses CV2024's
own validation split, reading directly from the manifest/image files.

Namespace: /data/exp015_cv2024_indomain_pipeline/stage2/

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage2_prune_surgery.py
"""
import json

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
PIPELINE_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
RAW_DIR = f"{PIPELINE_DIR}/raw"
VAL_DIR = f"{RAW_DIR}/Dataset/validation"
STUDENT_CHECKPOINT = f"{PIPELINE_DIR}/stage1/resnet18alpha075_cv2024_indomain_distilled_student.pt"
ARTIFACT_DIR = f"{PIPELINE_DIR}/stage2"
RESOLUTION = 224
BATCH_SIZE = 32
RATIO = 0.2
SAMPLES_PER_CLASS = 50

STEM_WIDTH = 24
STAGE_WIDTHS = {"layer1": 24, "layer2": 40, "layer3": 32, "layer4": 96}
PRUNE_LAYERS = ["layer2", "layer3", "layer4"]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "scikit-learn", "pandas", "numpy", "pillow", "thop")
)
app = modal.App("vce-exp015-stage2-prune-surgery-indomain", image=image)


def build_narrow_resnet_classes():
    import torch
    import torch.nn as nn

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

    class NarrowResNet(nn.Module):
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

    return BasicBlock, NarrowResNet


def slice_conv2d(old_conv, out_idx, in_idx):
    import torch
    import torch.nn as nn

    device = old_conv.weight.device
    out_idx_t = torch.as_tensor(list(out_idx), dtype=torch.long, device=device)
    in_idx_t = torch.as_tensor(list(in_idx), dtype=torch.long, device=device)
    new_conv = nn.Conv2d(
        in_channels=len(in_idx_t), out_channels=len(out_idx_t),
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=old_conv.bias is not None,
    ).to(device)
    w = old_conv.weight.data.index_select(0, out_idx_t)
    w = w.index_select(1, in_idx_t)
    new_conv.weight.data.copy_(w)
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data.index_select(0, out_idx_t))
    return new_conv


def slice_bn(old_bn, idx):
    import torch
    import torch.nn as nn

    device = old_bn.weight.device
    idx_t = torch.as_tensor(list(idx), dtype=torch.long, device=device)
    new_bn = nn.BatchNorm2d(
        num_features=len(idx_t), eps=old_bn.eps, momentum=old_bn.momentum,
        affine=old_bn.affine, track_running_stats=old_bn.track_running_stats,
    ).to(device)
    new_bn.weight.data.copy_(old_bn.weight.data.index_select(0, idx_t))
    new_bn.bias.data.copy_(old_bn.bias.data.index_select(0, idx_t))
    new_bn.running_mean.data.copy_(old_bn.running_mean.data.index_select(0, idx_t))
    new_bn.running_var.data.copy_(old_bn.running_var.data.index_select(0, idx_t))
    return new_bn


def prune_narrow_resnet(model, kept_indices: dict, stem_width: int):
    prev_idx = list(range(stem_width))
    for name in PRUNE_LAYERS:
        this_idx = kept_indices[name]
        layer = getattr(model, name)
        block0 = layer[0]
        block0.conv1 = slice_conv2d(block0.conv1, out_idx=this_idx, in_idx=prev_idx)
        block0.bn1 = slice_bn(block0.bn1, this_idx)
        block0.conv2 = slice_conv2d(block0.conv2, out_idx=this_idx, in_idx=this_idx)
        block0.bn2 = slice_bn(block0.bn2, this_idx)
        assert block0.downsample is not None
        block0.downsample[0] = slice_conv2d(block0.downsample[0], out_idx=this_idx, in_idx=prev_idx)
        block0.downsample[1] = slice_bn(block0.downsample[1], this_idx)
        block1 = layer[1]
        assert block1.downsample is None
        block1.conv1 = slice_conv2d(block1.conv1, out_idx=this_idx, in_idx=this_idx)
        block1.bn1 = slice_bn(block1.bn1, this_idx)
        block1.conv2 = slice_conv2d(block1.conv2, out_idx=this_idx, in_idx=this_idx)
        block1.bn2 = slice_bn(block1.bn2, this_idx)
        prev_idx = this_idx

    import torch
    import torch.nn as nn
    old_fc = model.fc
    fc_device = old_fc.weight.device
    last_idx_t = torch.as_tensor(prev_idx, dtype=torch.long, device=fc_device)
    new_fc = nn.Linear(in_features=len(last_idx_t), out_features=old_fc.out_features, bias=old_fc.bias is not None).to(fc_device)
    new_fc.weight.data.copy_(old_fc.weight.data.index_select(1, last_idx_t))
    if old_fc.bias is not None:
        new_fc.bias.data.copy_(old_fc.bias.data)
    model.fc = new_fc
    return model


def ranking_to_kept_indices(ranking: dict, layer_widths: dict, ratio: float) -> dict:
    kept = {}
    for name in PRUNE_LAYERS:
        width = layer_widths[name]
        order = list(ranking[name])
        assert len(order) == width
        assert sorted(order) == list(range(width))
        n_prune = int(round(ratio * width))
        prune_idx = set(order[:n_prune])
        kept[name] = sorted(i for i in range(width) if i not in prune_idx)
    return kept


def evaluate(model, val_loader, device):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return float(acc), float(macro_f1)


@app.function(gpu="T4", volumes={DATA_MOUNT: volume}, timeout=1800)
def run():
    import os

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from thop import profile
    from torch.utils.data import DataLoader, Dataset

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, NarrowResNet = build_narrow_resnet_classes()

    if not os.path.exists(STUDENT_CHECKPOINT):
        raise RuntimeError(f"Student checkpoint not found at {STUDENT_CHECKPOINT}")
    ckpt = torch.load(STUDENT_CHECKPOINT, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    stage_widths_list = ckpt["stage_widths"]
    assert stage_widths_list == [STAGE_WIDTHS["layer1"], STAGE_WIDTHS["layer2"], STAGE_WIDTHS["layer3"], STAGE_WIDTHS["layer4"]], \
        f"checkpoint stage_widths {stage_widths_list} != expected {STAGE_WIDTHS}"
    print(f"Loaded CV2024 in-domain alpha=0.75 checkpoint: best_epoch={ckpt.get('best_epoch')} "
          f"acc={ckpt.get('best_val_accuracy'):.4f} macro_f1={ckpt.get('best_val_macro_f1'):.4f}")

    val_manifest = pd.read_csv(f"{PIPELINE_DIR}/manifest_val.csv")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    eval_tf = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class CV2024ValDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(VAL_DIR, row["relpath"])
            img = Image.open(path).convert("RGB")
            return eval_tf(img), class_to_idx[row["class"]]

    val_loader = DataLoader(CV2024ValDataset(val_manifest), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    def load_full_width_student():
        m = NarrowResNet(STEM_WIDTH, stage_widths_list, num_classes=len(classes)).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m

    baseline_model = load_full_width_student()
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)
    baseline_acc, baseline_f1 = evaluate(baseline_model, val_loader, device)
    print(f"[sanity] full-width alpha=0.75 CV2024 in-domain baseline: acc={baseline_acc:.4f} macro_f1={baseline_f1:.4f}")

    profile_sample = pd.concat([g.sample(min(len(g), SAMPLES_PER_CLASS), random_state=0)
                                 for _, g in val_manifest.groupby("class")])
    pil_images, labels = [], []
    for _, r in profile_sample.iterrows():
        path = os.path.join(VAL_DIR, r["relpath"])
        pil_images.append(Image.open(path).convert("RGB"))
        labels.append(class_to_idx[r["class"]])
    print(f"PR+Fisher profile sample: {len(pil_images)} images")

    profile_model = load_full_width_student()
    profile_model.eval()
    for p in profile_model.parameters():
        p.requires_grad_(True)

    LAYER_GROUPS = ["layer1", "layer2", "layer3", "layer4"]
    fwd_activations = {name: [] for name in LAYER_GROUPS}
    bwd_grad_sq_sum = {name: None for name in LAYER_GROUPS}
    bwd_n_samples = {name: 0 for name in LAYER_GROUPS}

    def make_fwd_hook(name):
        def hook(module, inp, output):
            fwd_activations[name].append(output.mean(dim=[2, 3]).detach().cpu().numpy())
        return hook

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            per_sample_channel = (g ** 2).mean(dim=[2, 3])
            s = per_sample_channel.sum(dim=0).detach().cpu().numpy()
            if bwd_grad_sq_sum[name] is None:
                bwd_grad_sq_sum[name] = s
            else:
                bwd_grad_sq_sum[name] += s
            bwd_n_samples[name] += g.shape[0]
        return hook

    modules = {n: getattr(profile_model, n) for n in LAYER_GROUPS}
    hook_handles = []
    for name in LAYER_GROUPS:
        hook_handles.append(modules[name].register_forward_hook(make_fwd_hook(name)))
        hook_handles.append(modules[name].register_full_backward_hook(make_bwd_hook(name)))

    labels_t_all = torch.tensor(labels, dtype=torch.long)
    for start in range(0, len(pil_images), BATCH_SIZE):
        batch = pil_images[start:start + BATCH_SIZE]
        y = labels_t_all[start:start + BATCH_SIZE].to(device)
        x = torch.stack([eval_tf(im) for im in batch]).to(device)
        profile_model.zero_grad(set_to_none=True)
        logits = profile_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
    profile_model.zero_grad(set_to_none=True)
    for h in hook_handles:
        h.remove()

    ranking = {}
    channel_profile = {}
    for name in LAYER_GROUPS:
        X = np.concatenate(fwd_activations[name], axis=0)
        width = STAGE_WIDTHS[name]
        assert X.shape[1] == width
        channel_var = X.var(axis=0)
        v_c = channel_var / (channel_var.sum() + 1e-12)
        fisher_raw = bwd_grad_sq_sum[name] / max(bwd_n_samples[name], 1)
        ghat_c = fisher_raw / (fisher_raw.sum() + 1e-12)
        s_c = v_c + ghat_c
        order = np.argsort(s_c).tolist()
        ranking[name] = order
        channel_profile[name] = {
            "width": width, "v_c_pr_contribution_share": v_c.tolist(),
            "ghat_c_fisher_share": ghat_c.tolist(), "s_c_combined": s_c.tolist(),
            "ranking_ascending_least_important_first": order,
        }
        print(f"{name}: width={width} S_c range=[{s_c.min():.4f},{s_c.max():.4f}]")

    kept_indices = ranking_to_kept_indices(ranking, STAGE_WIDTHS, RATIO)
    for name in PRUNE_LAYERS:
        print(f"  {name}: {STAGE_WIDTHS[name]} -> {len(kept_indices[name])} kept channels (ratio={RATIO})")

    pruned = load_full_width_student()
    pruned = prune_narrow_resnet(pruned, kept_indices, STEM_WIDTH)
    pruned.to(device)
    pruned.eval()
    for p in pruned.parameters():
        p.requires_grad_(False)

    new_stage_widths = {
        "layer1": STAGE_WIDTHS["layer1"], "layer2": len(kept_indices["layer2"]),
        "layer3": len(kept_indices["layer3"]), "layer4": len(kept_indices["layer4"]),
    }

    acc, macro_f1 = evaluate(pruned, val_loader, device)
    print(f"post-surgery (no fine-tune): acc={acc:.4f} macro_f1={macro_f1:.4f}")

    dummy = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(device)
    macs, params = profile(pruned, inputs=(dummy,), verbose=False)
    flops_g = 2 * macs / 1e9
    params_m = params / 1e6
    print(f"params_M={params_m:.4f} flops_G_per_image_fwd={flops_g:.4f}")

    ckpt_out = {
        "state_dict": pruned.state_dict(), "stem_width": STEM_WIDTH,
        "stage_widths": [new_stage_widths["layer1"], new_stage_widths["layer2"], new_stage_widths["layer3"], new_stage_widths["layer4"]],
        "stage_widths_dict": new_stage_widths, "kept_indices": kept_indices, "classes": classes,
        "ranking_label": "pr_and_fisher_combined", "ratio": RATIO, "prune_layers": PRUNE_LAYERS,
        "source_checkpoint": STUDENT_CHECKPOINT, "params_M": params_m, "flops_G_per_image_fwd": flops_g,
        "post_surgery_val_accuracy": acc, "post_surgery_val_macro_f1": macro_f1,
    }
    ckpt_out_path = f"{ARTIFACT_DIR}/pruned_pr_and_fisher_r02_warmstart.pt"
    torch.save(ckpt_out, ckpt_out_path)
    volume.commit()
    print(f"Wrote checkpoint to volume at {ckpt_out_path}")

    with open(f"{ARTIFACT_DIR}/stage2_result.json", "w") as f:
        json.dump({
            "checkpoint": STUDENT_CHECKPOINT, "ratio": RATIO, "prune_layers": PRUNE_LAYERS,
            "n_val": len(val_manifest), "classes": classes,
            "full_width_baseline_on_this_pipeline": {"accuracy": baseline_acc, "macro_f1": baseline_f1},
            "kept_indices": kept_indices, "stage_widths": new_stage_widths,
            "post_surgery_accuracy": acc, "post_surgery_macro_f1": macro_f1,
            "params_M": params_m, "flops_G_per_image_fwd": flops_g,
        }, f, indent=2)
    volume.commit()

    return {
        "checkpoint": STUDENT_CHECKPOINT, "ratio": RATIO, "prune_layers": PRUNE_LAYERS,
        "n_val": len(val_manifest), "classes": classes,
        "full_width_baseline_on_this_pipeline": {"accuracy": baseline_acc, "macro_f1": baseline_f1},
        "stage_widths": new_stage_widths,
        "post_surgery_accuracy": acc, "post_surgery_macro_f1": macro_f1,
        "params_M": params_m, "flops_G_per_image_fwd": flops_g,
    }


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned call_id={call.object_id}")
