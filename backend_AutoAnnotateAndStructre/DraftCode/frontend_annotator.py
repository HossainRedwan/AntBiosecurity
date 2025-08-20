import argparse
from backend_auto_annotator import run_annotation

def main():
    parser = argparse.ArgumentParser(description="Auto Annotator for Ant Images Using YOLOv12")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    parser.add_argument("--model", required=True, help="Path to YOLOv12 pretrained model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    run_annotation(
        dataset_root=args.dataset,
        model_path=args.model,
        confidence=args.confidence
    )

if __name__ == "__main__":
    main()
