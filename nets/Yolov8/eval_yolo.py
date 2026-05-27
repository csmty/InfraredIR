import os
import glob
import shutil
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize YOLOv8 results on a validation set.")

    # Model
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the YOLO checkpoint."
    )

    # Input / label paths
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images to be evaluated."
    )
    parser.add_argument(
        "--gt_label_dir",
        type=str,
        required=True,
        help="Directory containing ground-truth label files."
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="./nets/Yolov8/infrared_val.yaml",
        help="Path to save the temporary data.yaml file."
    )

    # Inference settings
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for prediction."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation."
    )

    # Visualization / evaluation options
    parser.add_argument(
        "--vis_name",
        type=str,
        default="Visualization",
        help="Subfolder name for saved visualization results."
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="data",
        help="Subfolder name for saved evaluation results."
    )
    parser.add_argument(
        "--keep_tmp_dataset",
        action="store_true",
        help="Keep the temporary dataset after evaluation."
    )

    return parser.parse_args()


def build_tmp_dataset(save_dir, input_dir, gt_label_dir):
    """
    Build a temporary YOLO-style validation dataset:
        tmp_dataset/
            images/val/
            labels/val/
    """
    tmp_dataset = os.path.join(save_dir, "tmp_dataset")
    img_val_dir = os.path.join(tmp_dataset, "images", "val")
    label_val_dir = os.path.join(tmp_dataset, "labels", "val")

    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # Copy validation images
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for img_path in glob.glob(os.path.join(input_dir, ext)):
            shutil.copy(img_path, img_val_dir)

    # Copy ground-truth label files
    for label_path in glob.glob(os.path.join(gt_label_dir, "*.txt")):
        shutil.copy(label_path, label_val_dir)

    return tmp_dataset, img_val_dir, label_val_dir


def write_data_yaml(data_yaml_path, tmp_dataset):
    """Write a temporary YOLO dataset config file."""
    with open(data_yaml_path, "w") as f:
        f.write(
            f"""path: {tmp_dataset}
train: images/val
val: images/val
names:
  0: People
  1: Car
  2: Bus
  3: Motorcycle
  4: Lamp
  5: Truck
"""
        )


def save_metrics(results, model, save_dir):
    """Save evaluation metrics to a text file."""
    metrics_file = os.path.join(save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("=== YOLOv8 Evaluation Metrics ===\n\n")
        f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n\n")
        f.write(f"{'Class':12s} | {'Precision':>9s} | {'Recall':>9s} | {'mAP@0.5':>9s}\n")
        f.write("-" * 50 + "\n")
        for i, name in enumerate(model.names.values()):
            f.write(
                f"{name:12s} | "
                f"{results.box.p[i]:9.3f} | "
                f"{results.box.r[i]:9.3f} | "
                f"{results.box.maps[i]:9.3f}\n"
            )

    print(f"\nSaved per-class metrics to: {metrics_file}")


def run_evaluation(model, args, data_yaml):
    """Run YOLOv8 validation on the temporary dataset."""
    print("\n=== Running YOLOv8 Evaluation ===")
    results = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        project=args.save_dir,
        name=args.eval_name,
        exist_ok=True
    )
    return results


def run_visualization(model, args):
    """Run per-image prediction and save visualization results."""
    print("\n=== Running YOLOv8 Visualization ===")

    vis_dir = args.save_dir
    os.makedirs(vis_dir, exist_ok=True)

    all_images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        all_images.extend(glob.glob(os.path.join(args.input_dir, ext)))
    all_images = sorted(all_images)

    print(f"Found {len(all_images)} images.")

    for img_path in all_images:
        model.predict(
            source=img_path,
            conf=args.conf,
            save=True,
            project=vis_dir,
            name=args.vis_name,
            exist_ok=True,
            imgsz=args.imgsz
        )

    print(f"\nSaved visualization results to: {os.path.join(vis_dir, args.vis_name)}")
    if len(all_images) > 0:
        print(f"Example file: {os.path.join(vis_dir, args.vis_name, os.path.basename(all_images[0]))}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Build temporary validation dataset
    tmp_dataset, img_val_dir, label_val_dir = build_tmp_dataset(
        save_dir=args.save_dir,
        input_dir=args.input_dir,
        gt_label_dir=args.gt_label_dir
    )

    # Write dataset yaml
    write_data_yaml(args.data_yaml, tmp_dataset)

    # Load model
    model = YOLO(args.weights)

    # Run evaluation
    results = run_evaluation(model, args, args.data_yaml)

    # Save metrics
    save_metrics(results, model, args.save_dir)

    # Clean temporary dataset if needed
    if not args.keep_tmp_dataset and os.path.exists(tmp_dataset):
        shutil.rmtree(tmp_dataset)
        print(f"Removed temporary dataset: {tmp_dataset}")

    # Run visualization
    run_visualization(model, args)


if __name__ == "__main__":
    main()