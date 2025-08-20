import os
from pathlib import Path
from PIL import Image
from auto_annotation_yolov12 import Auto_Annotator, load_model, load_inference_image, process_predictions

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def is_image_file(file):
    return file.suffix.lower() in SUPPORTED_EXTENSIONS

def annotate_species_folder(model, species_path, conf_threshold=0.5):
    species_name = species_path.name
    image_files = [img for img in species_path.rglob("*") if img.is_file() and is_image_file(img)]
    if not image_files:
        print(f"[!] No images found in {species_path}")
        return

    annotation_dir = species_path / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        if "annotations" in img_path.parts:
            continue  # Skip annotation folder

        try:
            image, preds = load_inference_image(model, str(img_path))
            object_data = process_predictions(preds, conf_threshold, class_label=species_name)

            if not object_data:
                continue

            image_dims = [image.size[1], image.size[0], 3]  # height, width, channels
            annotator = Auto_Annotator(
                dataset_dir=str(species_path),
                annotation_path=str(annotation_dir),
                image_name=img_path.name,
                object_detection_data=object_data,
                image_dims=image_dims,
                confidence_threshold=conf_threshold
            )
            annotator.write_txt_annotation()
            print(f"[✓] Annotated {img_path.name}")
        except Exception as e:
            print(f"[!] Failed on {img_path.name}: {e}")

def run_annotation(dataset_root: str, model_path: str, confidence=0.5):
    dataset_path = Path(dataset_root)
    model = load_model(model_path)

    for species_folder in dataset_path.iterdir():
        if species_folder.is_dir():
            print(f"\n[•] Annotating species: {species_folder.name}")
            annotate_species_folder(model, species_folder, confidence)
