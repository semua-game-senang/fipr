"""exp015 (in-domain restart) stage 0a: download Capsule Vision 2024's own
train/val Dataset.zip from figshare (article 26403469) directly into the
Modal volume, unzip, inspect the 10-class taxonomy + counts, and run a
train/val duplicate-filename/hash leakage sanity check against
training_data.xlsx / validation_data.xlsx. Does NOT re-split anything --
their train/ and validation/ folder boundary is used as-is.

The CV2024 test set is already present on the volume at
/data/cv2024_test_exp015/ (downloaded during the discarded first pass --
that's just data, reusable). This script only fetches train+val.

New namespace: /data/exp015_cv2024_indomain_pipeline/raw/{train,validation}/

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach experiments/15-cv2024-augmented-pipeline/training/stage0a_download_prep_data.py
"""
import json

import modal

VOLUME_NAME = "vce-dataset"
DATA_MOUNT = "/data"
ARTIFACT_DIR = f"{DATA_MOUNT}/exp015_cv2024_indomain_pipeline"
RAW_DIR = f"{ARTIFACT_DIR}/raw"
FIGSHARE_ARTICLE_ID = 26403469

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests", "pandas", "openpyxl", "tqdm"
)
app = modal.App("vce-exp015-stage0a-download-prep", image=image)


@app.function(volumes={DATA_MOUNT: volume}, timeout=7200, cpu=4)
def run():
    import hashlib
    import os
    import zipfile

    import pandas as pd
    import requests

    os.makedirs(RAW_DIR, exist_ok=True)

    marker = f"{ARTIFACT_DIR}/.download_complete"
    if not os.path.exists(marker):
        # Resolve the actual downloadable file URL via the figshare API
        # (article page URL itself is not a direct-download link).
        meta = requests.get(
            f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}", timeout=60
        ).json()
        files = meta["files"]
        print("figshare files:", [(f["name"], f["size"]) for f in files])
        zip_file = next(f for f in files if f["name"].lower().endswith(".zip"))
        download_url = zip_file["download_url"]
        zip_path = f"{RAW_DIR}/Dataset.zip"

        print(f"downloading {zip_file['name']} ({zip_file['size']} bytes) from {download_url}")
        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"downloaded to {zip_path}, size={os.path.getsize(zip_path)}")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        print("extracted")
        os.remove(zip_path)
        volume.commit()

        with open(marker, "w") as f:
            f.write("done")
        volume.commit()
    else:
        print("download already complete, skipping")

    # ---------------- inspect structure ----------------
    def find_dir(root, name_lower_substr):
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                if name_lower_substr in d.lower():
                    return os.path.join(dirpath, d)
        return None

    train_dir = find_dir(RAW_DIR, "training") or find_dir(RAW_DIR, "train")
    val_dir = find_dir(RAW_DIR, "validation") or find_dir(RAW_DIR, "val")
    print(f"train_dir={train_dir} val_dir={val_dir}")

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def class_counts(d):
        counts = {}
        for entry in sorted(os.listdir(d)):
            p = os.path.join(d, entry)
            if os.path.isdir(p):
                n = 0
                for _, _, fns in os.walk(p):
                    n += sum(1 for fn in fns if fn.lower().endswith(IMG_EXTS))
                counts[entry] = n
        return counts

    train_counts = class_counts(train_dir)
    val_counts = class_counts(val_dir)
    print("train_counts:", train_counts)
    print("val_counts:", val_counts)

    # ---------------- locate metadata xlsx ----------------
    xlsx_files = []
    for dirpath, _, filenames in os.walk(RAW_DIR):
        for fn in filenames:
            if fn.lower().endswith(".xlsx"):
                xlsx_files.append(os.path.join(dirpath, fn))
    print("xlsx_files found:", xlsx_files)

    train_xlsx = next((p for p in xlsx_files if "train" in os.path.basename(p).lower()), None)
    val_xlsx = next((p for p in xlsx_files if "valid" in os.path.basename(p).lower()), None)

    leakage_report = {"train_xlsx": train_xlsx, "val_xlsx": val_xlsx}

    def build_filename_index(d):
        """Fast: filenames + paths only, no hashing (avoid per-file reads over
        the whole ~53k-image pool -- hashing the full sets was too slow over
        the Modal volume network FS in a first attempt)."""
        idx = {}
        for cls in sorted(os.listdir(d)):
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            for dirpath, _, fns in os.walk(cls_dir):
                for fn in fns:
                    if not fn.lower().endswith(IMG_EXTS):
                        continue
                    idx[fn] = (cls, os.path.join(dirpath, fn))
        return idx

    train_idx = build_filename_index(train_dir)
    val_idx = build_filename_index(val_dir)

    common_names = set(train_idx) & set(val_idx)
    # Only hash the (expected to be small/zero) filename-overlap set to check
    # for exact content duplicates -- avoids hashing all ~53k images.
    hash_dupes = []
    for name in common_names:
        cls_t, fp_t = train_idx[name]
        cls_v, fp_v = val_idx[name]
        with open(fp_t, "rb") as fh:
            h_t = hashlib.md5(fh.read()).hexdigest()
        with open(fp_v, "rb") as fh:
            h_v = hashlib.md5(fh.read()).hexdigest()
        if h_t == h_v:
            hash_dupes.append(name)

    leakage_report.update({
        "n_train_files": len(train_idx),
        "n_val_files": len(val_idx),
        "n_common_filenames": len(common_names),
        "n_identical_content_dupes_among_common_filenames": len(hash_dupes),
        "identical_content_dupe_examples": hash_dupes[:20],
        "note": "Using CV2024's official train/validation folder boundary as-is (not re-split). "
                "Fast check: filenames are compared first (cheap), and only the filename-overlap "
                "set (if any) is content-hashed to confirm true duplicates -- informational only, "
                "split is NOT modified either way.",
    })
    print("LEAKAGE REPORT:", json.dumps(leakage_report, indent=2))

    def build_manifest(d, split_name):
        rows = []
        for cls in sorted(os.listdir(d)):
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            for dirpath, _, fns in os.walk(cls_dir):
                source = os.path.relpath(dirpath, cls_dir)
                source = "root" if source == "." else source
                for fn in fns:
                    if not fn.lower().endswith(IMG_EXTS):
                        continue
                    relpath = os.path.relpath(os.path.join(dirpath, fn), d)
                    rows.append({"split": split_name, "class": cls, "filename": fn,
                                 "source_subset": source, "relpath": relpath})
        return pd.DataFrame(rows)

    train_manifest = build_manifest(train_dir, "train")
    val_manifest = build_manifest(val_dir, "val")
    train_manifest.to_csv(f"{ARTIFACT_DIR}/manifest_train.csv", index=False)
    val_manifest.to_csv(f"{ARTIFACT_DIR}/manifest_val.csv", index=False)
    print(f"manifest_train rows={len(train_manifest)} manifest_val rows={len(val_manifest)}")

    os.makedirs(f"{ARTIFACT_DIR}/results", exist_ok=True)
    with open(f"{ARTIFACT_DIR}/results/leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    with open(f"{ARTIFACT_DIR}/results/class_counts.json", "w") as f:
        json.dump({"train": train_counts, "val": val_counts, "train_dir": train_dir, "val_dir": val_dir}, f, indent=2)
    volume.commit()

    return {
        "train_dir": train_dir, "val_dir": val_dir,
        "train_counts": train_counts, "val_counts": val_counts,
        "leakage_report": leakage_report,
    }


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"spawned call_id={call.object_id}")
