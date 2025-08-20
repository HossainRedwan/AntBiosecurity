from PIL import Image
import os
import torch
from ultralytics import YOLO
from pathlib import Path

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def is_image_file(file):
    return file.suffix.lower() in SUPPORTED_EXTENSIONS


class Auto_Annotator():
    def __init__(self, dataset_dir, annotation_path, image_name, object_detection_data, image_dims, confidence_threshold, class_id):
        self.dataset_dir = dataset_dir
        self.annotation_path = annotation_path
        self.image_name = image_name
        self.object_detection_data = object_detection_data
        self.image_dims = image_dims
        self.confidence_threshold = confidence_threshold
        self.class_id = class_id

    def get_labels_boxes(self):
        boxes = []
        labels = []
        for each in self.object_detection_data:
            labels.append(each[0])
            boxes.append(each[2].tolist())
        return boxes, labels

    def create_annotations_dir(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def write_txt_annotation(self):
        '''
        Function: writes YOLO-format txt annotation file for image
        '''
        boxes, labels = self.get_labels_boxes()
        full_folder_name = self.annotation_path
        self.create_annotations_dir(full_folder_name)

        annotation_lines = []
        for b in boxes:
            xmin, ymin, xmax, ymax = b
            img_w, img_h = self.image_dims[1], self.image_dims[0]
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w = (xmax - xmin) / img_w
            box_h = (ymax - ymin) / img_h
            annotation_lines.append(f"{self.class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        output_path = os.path.join(full_folder_name, self.image_name.rsplit('.', 1)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write("\n".join(annotation_lines))


def load_model(model_path="yolov12m.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("[✓] CUDA is available. Loading YOLOv12m model on GPU.")
    else:
        print("[!] CUDA is NOT available. Loading on CPU — slower performance expected.")

    model = YOLO(model_path)
    model.to(device)
    return model

def load_inference_image(model, img_path):
    image = Image.open(img_path).convert("RGB")
    results = model.predict(source=img_path, save=False, verbose=False)
    return image, results[0]

def process_predictions(preds, confidence_threshold, class_label):
    object_data = []
    if preds.boxes is None or len(preds.boxes) == 0:
        return object_data

    for box in preds.boxes:
        score = float(box.conf)
        if score >= confidence_threshold:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            object_data.append([class_label, score, xyxy])
    return object_data


def annotate_species_folder(model, species_path, class_id, conf_threshold=0.5):
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
                confidence_threshold=conf_threshold,
                class_id=class_id
            )
            annotator.write_txt_annotation()
            print(f"[✓] Annotated {img_path.name}")
        except Exception as e:
            print(f"[!] Failed on {img_path.name}: {e}")


def run_annotation(dataset_root: str, model_path: str, confidence=0.5):
    dataset_path = Path(dataset_root)
    model = load_model(model_path)

    # Step 1: Generate class mapping from folder names
    class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    class_mapping = {folder.name: idx for idx, folder in enumerate(sorted(class_folders))}
    print("\n[✓] Class Mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")

    # Step 2: Annotate each species folder
    for species_folder in class_folders:
        print(f"\n[•] Annotating species: {species_folder.name}")
        class_id = class_mapping[species_folder.name]
        annotate_species_folder(model, species_folder, class_id, confidence)

    return class_mapping


def run_annotation(dataset_root: str, model_path: str, confidence=0.5):
    dataset_path = Path(dataset_root)
    model = load_model(model_path)

    # Step 1: Generate class mapping from folder names
    class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    class_mapping = {folder.name: idx for idx, folder in enumerate(sorted(class_folders))}

    print("\n[✓] Class Mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")

    # Step 2: Annotate each species folder
    for species_folder in class_folders:
        print(f"\n[•] Annotating species: {species_folder.name}")
        class_id = class_mapping[species_folder.name]
        annotate_species_folder(model, species_folder, class_id, confidence)

    return class_mapping

