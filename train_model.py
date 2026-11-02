"""
Train + Evaluate + Inference (CPU-optimized)
Run this after face_mask_detection.py has prepared the dataset.
"""

import os
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(r"F:\Visao")
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

EPOCHS = 15
IMG_SIZE = 416
BATCH_SIZE = 8
MODEL_NAME = "yolov8n.pt"


def train():
    print("=" * 60)
    print("Training YOLOv8 Nano (CPU mode)")
    print(f"  Epochs: {EPOCHS}, ImgSize: {IMG_SIZE}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(RESULTS_DIR),
        name="train",
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        cache=True,
    )


def evaluate():
    print("\n" + "=" * 60)
    print("Evaluating on validation set")
    print("=" * 60)

    best_path = RESULTS_DIR / "train" / "weights" / "best.pt"
    model = YOLO(str(best_path))
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

    print("\n--- Results ---")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")
    for i, c in enumerate(CLASSES):
        if i < len(metrics.box.ap50):
            print(f"  AP@0.5 {c:<25} {metrics.box.ap50[i]:.4f}")
    return metrics


def inference():
    print("\n" + "=" * 60)
    print("Running inference on sample validation images")
    print("=" * 60)

    best_path = RESULTS_DIR / "train" / "weights" / "best.pt"
    model = YOLO(str(best_path))
    val_images = list((DATASET_DIR / "val" / "images").glob("*.png"))
    random.seed(42)
    samples = random.sample(val_images, min(9, len(val_images)))

    model.predict(
        source=[str(p) for p in samples],
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        project=str(RESULTS_DIR),
        name="predictions",
        exist_ok=True,
    )
    print(f"Saved {len(samples)} prediction images to {RESULTS_DIR / 'predictions'}")


def generate_report(metrics):
    report = f"""============================================================
FACE MASK DETECTION - EXPERIMENT REPORT
============================================================

1. PROBLEM
   Task: Object Detection - Face Mask Detection
   Goal: Detect if people wear masks correctly, incorrectly, or not at all
   Dataset: Face Mask Detection (Kaggle) - 853 images, 4072 annotations
   Classes: {', '.join(CLASSES)}
   Class distribution: with_mask=3232, without_mask=717, mask_weared_incorrect=123

2. METHODOLOGY
   Model: YOLOv8 Nano (pre-trained COCO, fine-tuned)
   Strategy: Transfer Learning
   Image size: {IMG_SIZE}x{IMG_SIZE}
   Epochs: {EPOCHS} (early stopping patience=5)
   Batch size: {BATCH_SIZE}
   Split: 80% train (682 images) / 20% val (171 images)
   Augmentation: YOLOv8 defaults (mosaic, flip, scale, HSV)
   Hardware: CPU

3. RESULTS
   mAP@0.5:      {metrics.box.map50:.4f}
   mAP@0.5:0.95: {metrics.box.map:.4f}
   Precision:     {metrics.box.mp:.4f}
   Recall:        {metrics.box.mr:.4f}
"""
    for i, c in enumerate(CLASSES):
        if i < len(metrics.box.ap50):
            report += f"   AP@0.5 {c:<25} {metrics.box.ap50[i]:.4f}\n"

    report += f"""
4. OUTPUT FILES
   {RESULTS_DIR / 'class_distribution.png'} - Class distribution
   {RESULTS_DIR / 'sample_images.png'} - Annotated samples
   {RESULTS_DIR / 'bbox_analysis.png'} - Bbox size analysis
   {RESULTS_DIR / 'train'} - Training curves, confusion matrix
   {RESULTS_DIR / 'val'} - Validation plots
   {RESULTS_DIR / 'predictions'} - Sample detections
   {RESULTS_DIR / 'train' / 'weights' / 'best.pt'} - Best model
"""

    path = RESULTS_DIR / "report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    train()
    metrics = evaluate()
    inference()
    generate_report(metrics)
    print("\nDONE!")
