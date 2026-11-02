"""
Face Mask Detection using YOLOv8
================================
Computer Vision Project - Deep Learning
Dataset: Face Mask Detection (Kaggle)
Classes: with_mask, without_mask, mask_weared_incorrect
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"F:\Visao")
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_NAME = "yolov8n.pt"  # nano model - fast training, good baseline


# ============================================================
# STEP 1: Convert Pascal VOC XML to YOLO format
# ============================================================
def parse_voc_xml(xml_path):
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format: class_id center_x center_y width height (normalized)
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp values to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        objects.append((CLASS_MAP[class_name], cx, cy, w, h))

    return objects


def prepare_dataset():
    """Convert VOC XML annotations to YOLO format and split into train/val."""
    print("=" * 60)
    print("STEP 1: Preparing dataset (VOC XML -> YOLO format)")
    print("=" * 60)

    # Create directory structure
    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get all annotation files
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    indices = list(range(len(xml_files)))
    random.shuffle(indices)
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_indices = set(indices[:split_idx])

    stats = {"train": {c: 0 for c in CLASSES}, "val": {c: 0 for c in CLASSES}}
    skipped = 0

    for i, xml_file in enumerate(xml_files):
        split = "train" if i in train_indices else "val"
        img_name = xml_file.stem + ".png"
        img_src = IMAGES_DIR / img_name

        if not img_src.exists():
            skipped += 1
            continue

        objects = parse_voc_xml(xml_file)
        if not objects:
            skipped += 1
            continue

        # Copy image
        shutil.copy2(img_src, DATASET_DIR / split / "images" / img_name)

        # Write YOLO label file
        label_path = DATASET_DIR / split / "labels" / (xml_file.stem + ".txt")
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in objects:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                stats[split][CLASSES[cls_id]] += 1

    # Print statistics
    print(f"\nDataset split (seed={RANDOM_SEED}):")
    print(f"  Train images: {len(list((DATASET_DIR / 'train' / 'images').glob('*')))}")
    print(f"  Val images:   {len(list((DATASET_DIR / 'val' / 'images').glob('*')))}")
    print(f"  Skipped:      {skipped}")

    print("\nClass distribution:")
    print(f"  {'Class':<25} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for cls in CLASSES:
        total = stats["train"][cls] + stats["val"][cls]
        print(f"  {cls:<25} {stats['train'][cls]:>8} {stats['val'][cls]:>8} {total:>8}")

    return stats


def create_data_yaml():
    """Create the YOLO data configuration file."""
    yaml_path = DATASET_DIR / "data.yaml"
    content = f"""path: {DATASET_DIR}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\nCreated data.yaml at {yaml_path}")
    return yaml_path


# ============================================================
# STEP 2: Exploratory Data Analysis
# ============================================================
def plot_class_distribution(stats):
    """Plot class distribution for train and val sets."""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    for idx, split in enumerate(["train", "val"]):
        values = [stats[split][c] for c in CLASSES]
        bars = axes[idx].bar(CLASSES, values, color=colors, edgecolor="black", linewidth=0.5)
        axes[idx].set_title(f"{split.capitalize()} Set", fontsize=14, fontweight="bold")
        axes[idx].set_ylabel("Number of annotations")
        axes[idx].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                          str(val), ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Class Distribution - Face Mask Detection", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved class_distribution.png")


def plot_sample_images():
    """Show sample images with bounding boxes from the dataset."""
    sample_xmls = random.sample(list(ANNOTATIONS_DIR.glob("*.xml")), min(6, len(list(ANNOTATIONS_DIR.glob("*.xml")))))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    color_map = {"with_mask": "#2ecc71", "without_mask": "#e74c3c", "mask_weared_incorrect": "#f39c12"}

    for ax, xml_file in zip(axes.flat, sample_xmls):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_name = root.find("filename").text
        img_path = IMAGES_DIR / img_name

        if not img_path.exists():
            continue

        img = Image.open(img_path)
        ax.imshow(img)

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            color = color_map.get(cls_name, "white")
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(xmin, ymin - 4, cls_name.replace("_", " "),
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        ax.set_title(img_name, fontsize=10)
        ax.axis("off")

    fig.suptitle("Sample Images with Annotations", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sample_images.png")


def plot_bbox_analysis():
    """Analyze bounding box sizes."""
    widths, heights, areas = [], [], []
    for xml_file in ANNOTATIONS_DIR.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            w = int(bbox.find("xmax").text) - int(bbox.find("xmin").text)
            h = int(bbox.find("ymax").text) - int(bbox.find("ymin").text)
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(widths, bins=40, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].set_title("Bounding Box Widths", fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")

    axes[1].hist(heights, bins=40, color="#e67e22", edgecolor="black", alpha=0.7)
    axes[1].set_title("Bounding Box Heights", fontweight="bold")
    axes[1].set_xlabel("Height (pixels)")

    axes[2].scatter(widths, heights, alpha=0.3, s=10, color="#9b59b6")
    axes[2].set_title("Width vs Height", fontweight="bold")
    axes[2].set_xlabel("Width (pixels)")
    axes[2].set_ylabel("Height (pixels)")

    fig.suptitle("Bounding Box Size Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bbox_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved bbox_analysis.png")


# ============================================================
# STEP 3: Train YOLOv8
# ============================================================
def train_model(data_yaml):
    """Train YOLOv8 model."""
    print("\n" + "=" * 60)
    print("STEP 3: Training YOLOv8")
    print("=" * 60)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(RESULTS_DIR),
        name="train",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    return results


# ============================================================
# STEP 4: Evaluate model
# ============================================================
def evaluate_model():
    """Run validation and generate metrics."""
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

    best_model_path = RESULTS_DIR / "train" / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"ERROR: Best model not found at {best_model_path}")
        return None

    model = YOLO(str(best_model_path))

    metrics = model.val(
        data=str(DATASET_DIR / "data.yaml"),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(RESULTS_DIR),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n--- Validation Results ---")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")

    print("\n  Per-class AP@0.5:")
    for i, cls_name in enumerate(CLASSES):
        if i < len(metrics.box.ap50):
            print(f"    {cls_name:<25} {metrics.box.ap50[i]:.4f}")

    return metrics


# ============================================================
# STEP 5: Inference on sample images
# ============================================================
def run_inference():
    """Run inference on validation images and save visualizations."""
    print("\n" + "=" * 60)
    print("STEP 5: Inference on sample images")
    print("=" * 60)

    best_model_path = RESULTS_DIR / "train" / "weights" / "best.pt"
    model = YOLO(str(best_model_path))

    val_images = list((DATASET_DIR / "val" / "images").glob("*.png"))
    sample_images = random.sample(val_images, min(9, len(val_images)))

    results = model.predict(
        source=[str(p) for p in sample_images],
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        project=str(RESULTS_DIR),
        name="predictions",
        exist_ok=True,
    )

    print(f"Saved predictions for {len(sample_images)} images")
    return results


# ============================================================
# STEP 6: Generate summary report
# ============================================================
def generate_report(metrics):
    """Generate a text summary of the experiment."""
    print("\n" + "=" * 60)
    print("STEP 6: Generating report")
    print("=" * 60)

    report = []
    report.append("=" * 60)
    report.append("FACE MASK DETECTION - EXPERIMENT REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("1. PROBLEM DESCRIPTION")
    report.append("-" * 40)
    report.append("Task: Object Detection - Face Mask Detection")
    report.append("Objective: Detect whether people are wearing masks correctly,")
    report.append("incorrectly, or not wearing masks at all.")
    report.append(f"Dataset: Face Mask Detection (Kaggle) - 853 images")
    report.append(f"Classes: {', '.join(CLASSES)}")
    report.append(f"Total annotations: 4072")
    report.append("")
    report.append("2. METHODOLOGY")
    report.append("-" * 40)
    report.append(f"Model: YOLOv8 Nano (pre-trained on COCO)")
    report.append(f"Training strategy: Transfer Learning (fine-tuning)")
    report.append(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    report.append(f"Epochs: {EPOCHS}")
    report.append(f"Batch size: {BATCH_SIZE}")
    report.append(f"Train/Val split: {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%")
    report.append(f"Optimizer: SGD (default YOLOv8)")
    report.append(f"Data augmentation: YOLOv8 default (mosaic, flip, scale, HSV)")
    report.append("")
    report.append("3. RESULTS")
    report.append("-" * 40)

    if metrics:
        report.append(f"mAP@0.5:      {metrics.box.map50:.4f}")
        report.append(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        report.append(f"Precision:     {metrics.box.mp:.4f}")
        report.append(f"Recall:        {metrics.box.mr:.4f}")
        report.append("")
        report.append("Per-class AP@0.5:")
        for i, cls_name in enumerate(CLASSES):
            if i < len(metrics.box.ap50):
                report.append(f"  {cls_name:<25} {metrics.box.ap50[i]:.4f}")

    report.append("")
    report.append("4. OUTPUT FILES")
    report.append("-" * 40)
    report.append(f"Results directory: {RESULTS_DIR}")
    report.append("  - class_distribution.png: Class distribution chart")
    report.append("  - sample_images.png: Sample annotated images")
    report.append("  - bbox_analysis.png: Bounding box size analysis")
    report.append("  - train/: Training curves, confusion matrix, etc.")
    report.append("  - val/: Validation metrics and plots")
    report.append("  - predictions/: Sample inference results")
    report.append("  - train/weights/best.pt: Best trained model")
    report.append("")

    report_text = "\n".join(report)
    report_path = RESULTS_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {report_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Face Mask Detection using YOLOv8")
    print("=" * 60)
    print()

    # Step 1: Prepare dataset
    stats = prepare_dataset()
    data_yaml = create_data_yaml()

    # Step 2: EDA
    plot_class_distribution(stats)
    plot_sample_images()
    plot_bbox_analysis()

    # Step 3: Train
    train_model(data_yaml)

    # Step 4: Evaluate
    metrics = evaluate_model()

    # Step 5: Inference
    run_inference()

    # Step 6: Report
    generate_report(metrics)

    print("\n" + "=" * 60)
    print("DONE! All results saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
