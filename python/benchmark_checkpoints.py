"""
Checkpoint Benchmark — compare multiple .pth files on synthetic test data.

Usage:
    python notebooks/benchmark_checkpoints.py          # from repo root
    python benchmark_checkpoints.py                    # from notebooks/ dir
    (also works in Google Colab unchanged)

Requirements: gdown, pycocotools, torchmetrics, timm, opencv-python-headless
"""

import gc
import json
import os
import sys
import time
import contextlib
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR   = Path(__file__).resolve().parent      # notebooks/
REPO_DIR     = SCRIPT_DIR.parent                    # symbol-detection/
PYTHON_DIR   = REPO_DIR / "python"

CHECKPOINTS_DIR  = REPO_DIR / "bm_checkpoints"
TEST_DATASET_DIR = REPO_DIR / "bm_dataset"
RESULTS_DIR      = REPO_DIR / "bm_results"


CHECKPOINTS = [
    {
        "label":     "No CIoU Loss · epoch 500",
        "filename":  "ckpt_v2clean_epoch500.pth",
        "gdrive_id": "1aEIEb1OeJPC3y3trYyU9Y7gowZ_jANjI",
    },
    # {
    #     "label":     "v5 · epoch 260",
    #     "filename":  "ckpt_v5_epoch260.pth",
    #     "gdrive_id": "14Izos7SngzVjaIEXS4fJWRPeXsUAq7OH",
    # },
    # {
    #     "label":     "v6 · epoch 160",
    #     "filename":  "ckpt_v6_epoch160.pth",
    #     "gdrive_id": "1sbbUBaEyg3jJCTyzSbsZo0R_VzD6VpEz",
    # },
    # {
    #     "label":     "v6 · epoch 120",
    #     "filename":  "ckpt_v6_epoch120.pth",
    #     "gdrive_id": "1E29GeemAeqP-hS6gFJOOUKvOqDkuGP6j",
    # },
    # {
    #     "label":     "v6 · epoch 260",
    #     "filename":  "ckpt_v6_epoch260.pth",
    #     "gdrive_id": "1QlsjuvHYTe8SyXcyKLChBz0ma3fWtn10",
    # },
    {
        "label": "With CIoU Loss · epoch 300",
        "filename": "ckpt_v6_epoch300.pth",
        "gdrive_id": "14bJxsvN47hqXkAR50zGVXm1eH57TGGSO",
    },
    # {
    #     "label":     "v2-clean · epoch 170",
    #     "filename":  "ckpt_v2clean_epoch170.pth",
    #     "gdrive_id": "1YQO4vkcezYOM0Mj1JUyRidWRzM12n5oj",
    # },

]

# ─── Benchmark settings ───────────────────────────────────────────────────────
NUM_TEST_IMAGES = 30
CONF_THRESHOLD  = 0.50
NUM_CLASSES     = 8       # 7 symbols + 1 background
RANDOM_SEED     = 42


# ─── 0. Setup ─────────────────────────────────────────────────────────────────
def setup():
    print("=== Setup ===")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add source packages to import path
    for sub in ["src", "floor-grid/src", "effects/src"]:
        p = str(PYTHON_DIR / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    # Generator uses relative data paths — must run from python/
    os.chdir(PYTHON_DIR)
    print(f"Working dir : {os.getcwd()}")
    print(f"Repo        : {REPO_DIR}")
    print(f"Checkpoints : {CHECKPOINTS_DIR}")
    print(f"Test dataset: {TEST_DATASET_DIR}\n")

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not found — install it with:  uv pip install gdown")
        sys.exit(1)


def download_checkpoints():
    import gdown

    print("=== Downloading checkpoints ===")
    for ck in CHECKPOINTS:
        dest = CHECKPOINTS_DIR / ck["filename"]
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  ✓ Already cached: {ck['filename']}")
            continue
        print(f"  ↓ {ck['label']} …")
        gdown.download(id=ck["gdrive_id"], output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  ✓ {dest.name} ({dest.stat().st_size / 1_048_576:.0f} MB)")
        else:
            print(f"  ✗ Download failed for {ck['label']}")
    print()


def generate_test_dataset() -> dict:
    from symbol_detection.dataset.generator import COCODatasetGenerator
    import random as _random

    print(f"=== Generating {NUM_TEST_IMAGES} test images (seed={RANDOM_SEED}) ===")
    _random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    gen = COCODatasetGenerator(
        output_dir=str(TEST_DATASET_DIR),
        symbols_dir="data/electrical-symbols",
        distractor_dir="data/furnitures-and-other",
    )
    coco_gt = gen.generate_dataset(
        num_images=NUM_TEST_IMAGES,
        rows=(20, 70),
        cols=(20, 70),
        cell_size=(8, 60),
        symbols_per_room=(0, 3),
        num_distractors_per_room=(0, 1),
        scale_range=(0.8, 1.2),
        discrete_rotations=[0, 90, 180, 270],
        apply_symbol_effects=False,
        apply_image_effects=True,
    )
    gen.save_annotations()

    print(f"Generated {NUM_TEST_IMAGES} images, "
          f"{len(coco_gt['annotations'])} ground-truth annotations.\n")
    return coco_gt


def load_model(checkpoint_path: Path, device: str):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        num_classes=NUM_CLASSES,
        trainable_backbone_layers=4,
    )
    ckpt  = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(model, coco_gt: dict, device: str) -> list:
    images_dir = TEST_DATASET_DIR / "images"
    predictions = []

    with torch.no_grad():
        for img_info in coco_gt["images"]:
            img_path = images_dir / img_info["file_name"]
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            h, w = image.shape[:2]
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            canvas = np.zeros((512, 512, 3), dtype=np.uint8)
            y_off = (512 - new_h) // 2
            x_off = (512 - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
            output = model([tensor.to(device)])[0]

            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < CONF_THRESHOLD:
                    continue
                x1 = max(0.0, min((box[0] - x_off) / scale, w))
                y1 = max(0.0, min((box[1] - y_off) / scale, h))
                x2 = max(0.0, min((box[2] - x_off) / scale, w))
                y2 = max(0.0, min((box[3] - y_off) / scale, h))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                predictions.append({
                    "image_id":    img_info["id"],
                    "category_id": int(label),
                    "bbox":        [float(x1), float(y1), float(bw), float(bh)],
                    "score":       float(score),
                })

    return predictions


def coco_eval(coco_gt: dict, predictions: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_gt, f)
        gt_file = f.name

    with contextlib.redirect_stdout(io.StringIO()):
        coco_obj  = COCO(gt_file)
        coco_dt   = coco_obj.loadRes(predictions) if predictions else coco_obj.loadRes([])
        ev        = COCOeval(coco_obj, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    os.unlink(gt_file)

    s = ev.stats
    return {"mAP": s[0], "AP50": s[1], "AP75": s[2], "mAR": s[8]}


def per_class_ap50(coco_gt: dict, preds: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt["categories"]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_gt, f)
        gt_file = f.name

    with contextlib.redirect_stdout(io.StringIO()):
        coco_obj  = COCO(gt_file)
        coco_dt   = coco_obj.loadRes(preds) if preds else coco_obj.loadRes([])
        ev        = COCOeval(coco_obj, coco_dt, "bbox")
        ev.params.iouThrs = np.array([0.50])
        ev.evaluate()
        ev.accumulate()

    os.unlink(gt_file)

    precision = ev.eval["precision"]   # (1, 101, K, 4, 3)
    result = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p    = precision[0, :, k, 0, 2]
        ap50 = float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0
        result[cat_id_to_name.get(cat_id, f"id={cat_id}")] = ap50
    return result


def save_json(results: list, class_aps_all: dict):
    import datetime

    payload = {
        "run_at":        datetime.datetime.now().isoformat(timespec="seconds"),
        "num_test_images": NUM_TEST_IMAGES,
        "conf_threshold":  CONF_THRESHOLD,
        "checkpoints": [
            {
                "label":      r["label"],
                "n_dets":     r["n_dets"],
                "elapsed_s":  round(r["elapsed"], 1),
                "metrics":    {k: round(v, 4) for k, v in r["metrics"].items()},
                "per_class_ap50": {
                    k: round(v, 4) for k, v in class_aps_all.get(r["label"], {}).items()
                },
            }
            for r in results
        ],
    }
    out = RESULTS_DIR / "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {out}")
    return payload


def save_plot(results: list, class_aps_all: dict):
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    labels      = [r["label"] for r in results]
    metric_keys = ["mAP", "AP50", "AP75", "mAR"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    n   = len(labels)
    x   = np.arange(n)
    bar_w = 0.18

    # Collect class names (union across all checkpoints)
    all_classes = []
    for ck_label in labels:
        for cls in class_aps_all.get(ck_label, {}):
            if cls not in all_classes:
                all_classes.append(cls)
    all_classes = sorted(all_classes)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(max(10, n * 2.2), 12),
        gridspec_kw={"height_ratios": [1.2, 1.8]},
    )
    fig.patch.set_facecolor("#F8F9FA")

    ax1.set_facecolor("#F8F9FA")
    for i, (mk, col) in enumerate(zip(metric_keys, colors)):
        vals = [r["metrics"][mk] for r in results]
        bars = ax1.bar(x + i * bar_w, vals, bar_w, label=mk, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )

    best_idx = max(range(len(results)), key=lambda i: results[i]["metrics"]["mAP"])
    ax1.axvspan(best_idx - 0.15, best_idx + 4 * bar_w + 0.05, alpha=0.07, color="gold")

    ax1.set_xticks(x + bar_w * 1.5)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score")
    ax1.set_title("Checkpoint Comparison — mAP / AP50 / AP75 / mAR", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.set_facecolor("#F8F9FA")
    class_x   = np.arange(len(all_classes))
    ck_colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    ck_bar_w  = 0.8 / max(n, 1)

    for i, (r, col) in enumerate(zip(results, ck_colors)):
        vals = [class_aps_all.get(r["label"], {}).get(cls, 0.0) for cls in all_classes]
        ax2.bar(class_x + i * ck_bar_w, vals, ck_bar_w, label=r["label"],
                color=col, alpha=0.82)

    ax2.set_xticks(class_x + ck_bar_w * (n - 1) / 2)
    ax2.set_xticklabels(all_classes, rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("AP50")
    ax2.set_title("Per-class AP50 by Checkpoint", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=2.5)

    out = RESULTS_DIR / "benchmark_plot.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved     → {out}")


def main():
    setup()
    download_checkpoints()
    coco_gt = generate_test_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    results = []

    for ck in CHECKPOINTS:
        ckpt_path = CHECKPOINTS_DIR / ck["filename"]
        if not ckpt_path.exists() or ckpt_path.stat().st_size < 1_000_000:
            print(f"  ✗ Skipping {ck['label']} — file missing or too small.\n")
            continue

        print(f"=== {ck['label']} ===")
        t0 = time.time()

        try:
            model   = load_model(ckpt_path, device)
            preds   = run_inference(model, coco_gt, device)
            metrics = coco_eval(coco_gt, preds)
            elapsed = time.time() - t0

            results.append({
                "label":   ck["label"],
                "metrics": metrics,
                "preds":   preds,
                "n_dets":  len(preds),
                "elapsed": elapsed,
            })

            print(f"  mAP={metrics['mAP']:.3f}  AP50={metrics['AP50']:.3f}  "
                  f"AP75={metrics['AP75']:.3f}  mAR={metrics['mAR']:.3f}  "
                  f"({elapsed:.0f}s, {len(preds)} detections)")

        except Exception as exc:
            import traceback
            print(f"  ✗ Error: {exc}")
            traceback.print_exc()
        finally:
            if "model" in dir():
                del model
            torch.cuda.empty_cache()
            gc.collect()

        print()

    if not results:
        print("No results to display.")
        return

    W = 70
    print("=" * W)
    print("BENCHMARK RESULTS")
    print("=" * W)
    print(f"{'Checkpoint':<26}  {'mAP':>6}  {'AP50':>6}  {'AP75':>6}  {'mAR':>6}  {'Dets':>6}")
    print("-" * W)

    best = max(results, key=lambda r: r["metrics"]["mAP"])

    for r in results:
        m      = r["metrics"]
        marker = "  ◄ BEST" if r["label"] == best["label"] else ""
        print(f"{r['label']:<26}  {m['mAP']:6.3f}  {m['AP50']:6.3f}  "
              f"{m['AP75']:6.3f}  {m['mAR']:6.3f}  {r['n_dets']:6d}{marker}")

    print("=" * W)
    bm = best["metrics"]
    print(f"\nBest checkpoint : {best['label']}")
    print(f"  mAP={bm['mAP']:.3f}  AP50={bm['AP50']:.3f}  "
          f"AP75={bm['AP75']:.3f}  mAR={bm['mAR']:.3f}")

    print("\n--- Per-class AP50 ---")
    class_aps_all = {}
    for r in results:
        class_aps_all[r["label"]] = per_class_ap50(coco_gt, r["preds"])

    for name, ap in sorted(class_aps_all[best["label"]].items(), key=lambda x: -x[1]):
        bar = "█" * int(ap * 30)
        print(f"  {name:<35} {ap:.3f}  {bar}")

    print()
    save_json(results, class_aps_all)
    save_plot(results, class_aps_all)


if __name__ == "__main__":
    main()
