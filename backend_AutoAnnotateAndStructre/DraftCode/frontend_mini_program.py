from backend_auto_annotator import run_annotation

def main():
    print("=== Ant Auto Annotator (YOLOv12) ===\n")

    dataset_path = input("Enter the full path to your dataset folder: ").strip()

    # Hardcoded values
    model_path = "./models/yolov12_ant.pt"  # You can change this to your actual path
    confidence_threshold = 0.5

    print(f"\n[INFO] Using model: {model_path}")
    print(f"[INFO] Confidence threshold: {confidence_threshold}")
    print(f"[INFO] Dataset: {dataset_path}\n")

    run_annotation(
        dataset_root=dataset_path,
        model_path=model_path,
        confidence=confidence_threshold
    )

if __name__ == "__main__":
    main()
