"""
Experimental Pipeline - Face Mask Detection with YOLOv8

Methodology: controlled-variable experiments.
  - Define baseline configuration (YOLOv8 defaults)
  - Vary ONE parameter per experiment
  - Compare all experiments on the same validation set
  - Select final config based on metrics + trade-offs

Usage:
  python run_experiments.py           # run all experiments
  python run_experiments.py --final   # train final model with best config
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(r"F:\Visao")
DATASET_DIR = BASE_DIR / "dataset"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

# ── Baseline: YOLOv8 recommended defaults ──────────────────────
BASELINE = {
    "model": "yolov8n.pt",
    "epochs": 15,
    "imgsz": 640,
    "batch": 8,
    "lr0": 0.01,
    "mosaic": 1.0,
}

# ── Experiments: each changes exactly ONE variable ─────────────
EXPERIMENTS = [
    {
        "name": "exp0_baseline",
        "desc": "Baseline (YOLOv8n, 640, defaults)",
        "var": "-",
        "changes": {},
    },
    {
        "name": "exp1_model_small",
        "desc": "Model: YOLOv8s (Small)",
        "var": "model",
        "changes": {"model": "yolov8s.pt"},
    },
    {
        "name": "exp2_imgsz_416",
        "desc": "Resolution: 416x416",
        "var": "imgsz",
        "changes": {"imgsz": 416},
    },
    {
        "name": "exp3_imgsz_320",
        "desc": "Resolution: 320x320",
        "var": "imgsz",
        "changes": {"imgsz": 320},
    },
    {
        "name": "exp4_lr_low",
        "desc": "Learning rate: 0.001",
        "var": "lr0",
        "changes": {"lr0": 0.001},
    },
    {
        "name": "exp5_no_mosaic",
        "desc": "Augmentation: mosaic disabled",
        "var": "mosaic",
        "changes": {"mosaic": 0.0},
    },
]


def run_single(exp):
    cfg = {**BASELINE, **exp["changes"]}
    name = exp["name"]
    out_dir = EXPERIMENTS_DIR / name

    result_file = out_dir / "experiment_result.json"
    if result_file.exists():
        print(f"\n[SKIP] {name} — already done (delete folder to re-run)")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f" {name}: {exp['desc']}")
    print(f" Changed: {exp['var']} | Config: {cfg}")
    print(f"{'='*70}")

    model = YOLO(cfg["model"])
    t0 = time.time()
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        lr0=cfg["lr0"],
        mosaic=cfg["mosaic"],
        project=str(EXPERIMENTS_DIR),
        name=name,
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        cache=True,
    )
    elapsed = time.time() - t0

    best_pt = out_dir / "weights" / "best.pt"
    val_model = YOLO(str(best_pt))
    metrics = val_model.val(
        data=str(DATASET_DIR / "data.yaml"),
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        project=str(EXPERIMENTS_DIR),
        name=f"{name}_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    result = {
        "name": name,
        "desc": exp["desc"],
        "var": exp["var"],
        "model": cfg["model"],
        "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"],
        "lr0": cfg["lr0"],
        "mosaic": cfg["mosaic"],
        "mAP50": round(metrics.box.map50, 4),
        "mAP50_95": round(metrics.box.map, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "train_min": round(elapsed / 60, 1),
    }
    for i, c in enumerate(CLASSES):
        if i < len(metrics.box.ap50):
            result[f"AP50_{c}"] = round(metrics.box.ap50[i], 4)

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Plots ──────────────────────────────────────────────────────

def generate_plots(results):
    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    labels = [f"E{i}" for i in range(len(results))]
    descs = [r["desc"] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # 1 — Overall metrics (4 bar charts)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, key, title in zip(
        axes,
        ["mAP50", "mAP50_95", "precision", "recall"],
        ["mAP@50", "mAP@50-95", "Precision", "Recall"],
    ):
        vals = [r[key] for r in results]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.3f}",
                ha="center", fontsize=8,
            )
    fig.suptitle("Experiment Comparison — Overall Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2 — Per-class AP@50
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results))
    w = 0.25
    for i, c in enumerate(CLASSES):
        vals = [r.get(f"AP50_{c}", 0) for r in results]
        bars = ax.bar(x + i * w, vals, w, label=c)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AP@50")
    ax.legend()
    ax.set_title("Per-Class AP@50 by Experiment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots_dir / "perclass_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3 — Training time
    fig, ax = plt.subplots(figsize=(10, 5))
    times = [r["train_min"] for r in results]
    bars = ax.bar(labels, times, color=colors)
    for bar, v in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.0f}min", ha="center", fontsize=9,
        )
    ax.set_ylabel("Training Time (min)")
    ax.set_title("Training Time by Experiment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots_dir / "time_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4 — Accuracy vs Speed scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        ax.scatter(r["train_min"], r["mAP50"], s=120, c=[colors[i]], zorder=5)
        ax.annotate(
            f"{labels[i]}\n{r['desc']}", (r["train_min"], r["mAP50"]),
            textcoords="offset points", xytext=(8, 4), fontsize=7,
        )
    ax.set_xlabel("Training Time (min)")
    ax.set_ylabel("mAP@50")
    ax.set_title("Accuracy vs Training Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Legend file mapping E0..E5 to descriptions
    legend_path = plots_dir / "legend.txt"
    with open(legend_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(f"E{i} = {r['desc']}\n")

    print(f"Plots saved to {plots_dir}/")


# ── Text + CSV report ──────────────────────────────────────────

def generate_report(results):
    csv_path = EXPERIMENTS_DIR / "comparison.csv"
    fields = [
        "name", "desc", "var", "model", "imgsz", "lr0", "mosaic",
        "mAP50", "mAP50_95", "precision", "recall", "train_min",
        "AP50_with_mask", "AP50_without_mask", "AP50_mask_weared_incorrect",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    lines = [
        "=" * 95,
        "EXPERIMENTAL COMPARISON — Face Mask Detection with YOLOv8",
        "=" * 95,
        "",
        f"Baseline config: {BASELINE}",
        "Dataset: 853 images (682 train / 171 val), 3 classes",
        "Method: vary ONE parameter at a time from baseline",
        "",
    ]

    hdr = (f"{'#':<4} {'Description':<36} {'Var':<8} "
           f"{'mAP@50':>8} {'mAP@.5:.95':>11} {'Prec':>7} {'Rec':>7} {'Time':>7}")
    lines += [hdr, "-" * len(hdr)]

    best_map, best_name = 0, ""
    for i, r in enumerate(results):
        lines.append(
            f"E{i:<3} {r['desc']:<36} {r['var']:<8} "
            f"{r['mAP50']:>8.4f} {r['mAP50_95']:>11.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['train_min']:>6.0f}m"
        )
        if r["mAP50"] > best_map:
            best_map, best_name = r["mAP50"], r["name"]

    lines += ["", f">>> Best mAP@50: {best_name} ({best_map:.4f})", ""]

    lines.append("Per-class AP@50:")
    hdr2 = f"{'#':<4} {'with_mask':>12} {'without_mask':>14} {'incorrect':>12}"
    lines += [hdr2, "-" * len(hdr2)]
    for i, r in enumerate(results):
        lines.append(
            f"E{i:<3} {r.get('AP50_with_mask', 0):>12.4f} "
            f"{r.get('AP50_without_mask', 0):>14.4f} "
            f"{r.get('AP50_mask_weared_incorrect', 0):>12.4f}"
        )

    report = "\n".join(lines)
    report_path = EXPERIMENTS_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    return best_name


# ── Final model ────────────────────────────────────────────────

def train_final():
    csv_path = EXPERIMENTS_DIR / "comparison.csv"
    if not csv_path.exists():
        print("ERROR: run experiments first (python run_experiments.py)")
        sys.exit(1)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    best_row = max(rows, key=lambda r: float(r["mAP50"]))
    best_name = best_row["name"]

    result_file = EXPERIMENTS_DIR / best_name / "experiment_result.json"
    with open(result_file) as f:
        best = json.load(f)

    cfg = {k: best.get(k, BASELINE[k]) for k in BASELINE}
    cfg["epochs"] = 30

    print(f"\n{'='*70}")
    print(f" FINAL MODEL — config from {best_name}")
    print(f" Config: {cfg}  (epochs bumped to 30)")
    print(f"{'='*70}")

    model = YOLO(cfg["model"])
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=cfg["epochs"],
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        mosaic=float(cfg["mosaic"]),
        project=str(EXPERIMENTS_DIR),
        name="final",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        cache=True,
    )

    best_pt = EXPERIMENTS_DIR / "final" / "weights" / "best.pt"
    val_model = YOLO(str(best_pt))
    metrics = val_model.val(
        data=str(DATASET_DIR / "data.yaml"),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        project=str(EXPERIMENTS_DIR),
        name="final_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*70}")
    print(f" FINAL RESULTS")
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print(f"  Weights:      {best_pt}")
    print(f"{'='*70}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection — Experimental Pipeline")
    parser.add_argument("--final", action="store_true", help="Train final model with best config")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    if args.final:
        train_final()
    else:
        results = []
        for exp in EXPERIMENTS:
            r = run_single(exp)
            if r:
                results.append(r)

        if results:
            best = generate_report(results)
            generate_plots(results)
            print(f"\n{'='*70}")
            print(f" Done. Best config: {best}")
            print(f" Next step: python run_experiments.py --final")
            print(f"{'='*70}")
