import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def is_image_file(file):
    return file.suffix.lower() in SUPPORTED_EXTENSIONS

def normalize_species_name(name: str):
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")

class Auto_Annotator:
    def __init__(self, annotation_path, image_name, object_detection_data, image_dims, confidence_threshold, class_id):
        self.annotation_path = annotation_path
        self.image_name = image_name
        self.object_detection_data = object_detection_data
        self.image_dims = image_dims
        self.confidence_threshold = confidence_threshold
        self.class_id = class_id

    def get_labels_boxes(self):
        boxes, labels = [], []
        for each in self.object_detection_data:
            labels.append(each[0])
            boxes.append(each[2].tolist())
        return boxes, labels

    def create_annotations_dir(self):
        os.makedirs(self.annotation_path, exist_ok=True)

    def write_txt_annotation(self):
        boxes, _ = self.get_labels_boxes()
        self.create_annotations_dir()
        annotation_lines = []
        img_w, img_h = self.image_dims[1], self.image_dims[0]

        for b in boxes:
            xmin, ymin, xmax, ymax = b
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w = (xmax - xmin) / img_w
            box_h = (ymax - ymin) / img_h
            annotation_lines.append(f"{self.class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        output_path = os.path.join(self.annotation_path, self.image_name.rsplit('.', 1)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write("\n".join(annotation_lines))

    def write_xml_annotation(self):
        boxes, labels = self.get_labels_boxes()
        img_w, img_h, img_d = self.image_dims[1], self.image_dims[0], self.image_dims[2]

        root = Element('annotation')
        SubElement(root, 'folder').text = os.path.basename(self.annotation_path)
        SubElement(root, 'filename').text = self.image_name
        SubElement(root, 'path').text = self.annotation_path
        source = SubElement(root, 'source')
        SubElement(source, 'database').text = 'Unknown'

        size = SubElement(root, 'size')
        SubElement(size, 'width').text = str(img_w)
        SubElement(size, 'height').text = str(img_h)
        SubElement(size, 'depth').text = str(img_d)
        SubElement(root, 'segmented').text = '0'

        for b, lbl in zip(boxes, labels):
            obj = SubElement(root, 'object')
            SubElement(obj, 'name').text = lbl
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(b[0])
            SubElement(bndbox, 'ymin').text = str(b[1])
            SubElement(bndbox, 'xmax').text = str(b[2])
            SubElement(bndbox, 'ymax').text = str(b[3])

        xml_str = parseString(tostring(root)).toprettyxml(indent="  ")
        xml_path = os.path.join(self.annotation_path, self.image_name.rsplit('.', 1)[0] + ".xml")
        with open(xml_path, "w") as f:
            f.write(xml_str)

def load_model(model_path="yolov12m.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading YOLOv12m on {'GPU' if device.type=='cuda' else 'CPU'}...")
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

def annotate_and_move_species(model, species_path, species_name, class_id, autoannotated_root, conf_threshold=0.5):
    image_files = [img for img in species_path.rglob("*") if img.is_file() and is_image_file(img)]
    if not image_files:
        print(f"[!] No images found for {species_name}. Skipping...")
        return 0

    # Prepare destination folders
    species_folder = Path(autoannotated_root) / species_name
    img_dest = species_folder / f"{species_name}_Images"
    annot_dest = species_folder / f"{species_name}_Annotations"
    xml_txt = annot_dest / "XML_and_TXT"
    xml_only = annot_dest / "XML_Only"
    txt_only = annot_dest / "TXT_Only"

    for folder in [img_dest, xml_txt, xml_only, txt_only]:
        folder.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in image_files:
        try:
            image, preds = load_inference_image(model, str(img_path))
            object_data = process_predictions(preds, conf_threshold, class_label=species_name)
            if not object_data:
                continue

            image_dims = [image.size[1], image.size[0], 3]  # H,W,C
            annotator = Auto_Annotator(str(xml_txt), img_path.name, object_data, image_dims, conf_threshold, class_id)
            annotator.write_txt_annotation()
            annotator.write_xml_annotation()

            # Copy txt and xml to respective folders
            txt_file = xml_txt / (img_path.stem + ".txt")
            xml_file = xml_txt / (img_path.stem + ".xml")
            shutil.copy(txt_file, txt_only / txt_file.name)
            shutil.copy(xml_file, xml_only / xml_file.name)

            # Move image to images folder
            shutil.move(str(img_path), img_dest / img_path.name)
            count += 1
            print(f"[✓] Annotated {img_path.name}")
        except Exception as e:
            print(f"[!] Failed to process {img_path.name}: {e}")

    # Remove species folder after processing
    shutil.rmtree(species_path)
    return count

"""
def run_annotation(
    dataset_root: str,
    model_path: str,
    to_annotate_path: str,
    auto_annotated_path: str,
    mapping_file_path: str,
    confidence: float = 0.5
):
    dataset_path = Path(dataset_root)
    to_annotate = Path(to_annotate_path)
    auto_annotated = Path(auto_annotated_path)
    mapping_file = Path(mapping_file_path)

    if not mapping_file.exists():
        raise FileNotFoundError(
            f"[ERROR] master_class_mapping.csv does not exist at {mapping_file.resolve()}!"
        )

    # Load CSV instead of Excel
    mapping_df = pd.read_csv(mapping_file)
    species_map = {
        normalize_species_name(row["SpeciesName"]): int(row["ClassID"])
        for _, row in mapping_df.iterrows()
    }
    max_class_id = max(species_map.values()) if species_map else 0

    species_folders = [f for f in to_annotate.iterdir() if f.is_dir()]
    if not species_folders:
        print("[INFO] No species to annotate.")
        return

    model = load_model(model_path)
    total_images = 0
    new_species_added = []

    for species_folder in species_folders:
        raw_name = species_folder.name
        norm_name = normalize_species_name(raw_name)

        # Add new species to mapping if not found
        if norm_name not in species_map:
            max_class_id += 1
            species_map[norm_name] = max_class_id
            mapping_df = pd.concat(
                [mapping_df, pd.DataFrame([[max_class_id, raw_name]], columns=["ClassID", "SpeciesName"])]
            )
            new_species_added.append(raw_name)

        class_id = species_map[norm_name]
        print(f"\n[•] Processing species: {raw_name} (ClassID: {class_id})")

        count = annotate_and_move_species(
            model, species_folder, raw_name, class_id, auto_annotated, confidence
        )
        total_images += count

    # Save CSV instead of Excel
    mapping_df.to_csv(mapping_file, index=False)

    print("\n=== Summary Report ===")
    print(f"Total species processed: {len(species_folders)}")
    print(f"Total images annotated: {total_images}")
    print(f"New species added: {new_species_added if new_species_added else 'None'}")

"""
"""
def run_annotation( dataset_root: str, model_path: str, to_annotate_path: str, auto_annotated_path: str, mapping_file_path: str, confidence: float = 0.5 ):   
    dataset_path = Path(dataset_root)
    to_annotate = Path(to_annotate_path)
    auto_annotated = Path(auto_annotated_path)
    mapping_file = Path(mapping_file_path)

    if not mapping_file.exists():
        raise FileNotFoundError("[ERROR] master_class_mapping.xlsx does not exist in dataset root!")

    mapping_df = pd.read_excel(mapping_file)
    species_map = {
        normalize_species_name(row["SpeciesName"]): int(row["ClassID"])
        for _, row in mapping_df.iterrows()
    }
    max_class_id = max(species_map.values()) if species_map else 0

    species_folders = [f for f in to_annotate.iterdir() if f.is_dir()]
    if not species_folders:
        print("[INFO] No species to annotate.")
        return

    model = load_model(model_path)
    total_images = 0
    new_species_added = []

    for species_folder in species_folders:
        raw_name = species_folder.name
        norm_name = normalize_species_name(raw_name)

        # Add new species to mapping if not found
        if norm_name not in species_map:
            max_class_id += 1
            species_map[norm_name] = max_class_id
            mapping_df = pd.concat(
                [mapping_df, pd.DataFrame([[max_class_id, raw_name]], columns=["ClassID", "SpeciesName"])]
            )
            new_species_added.append(raw_name)

        class_id = species_map[norm_name]
        print(f"\n[•] Processing species: {raw_name} (ClassID: {class_id})")

        count = annotate_and_move_species(
            model, species_folder, raw_name, class_id, auto_annotated, confidence
        )
        total_images += count

    # Update Excel mapping
    mapping_df.to_excel(mapping_file, index=False)

    print("\n=== Summary Report ===")
    print(f"Total species processed: {len(species_folders)}")
    print(f"Total images annotated: {total_images}")
    print(f"New species added: {new_species_added if new_species_added else 'None'}")


"""
"""
def run_annotation(dataset_root: str, model_path: str, confidence=0.5,to_annotate_path, auto_annotated_path, mapping_file_path  ):

def run_annotation(
    dataset_root: str, 
    model_path: str, 
    to_annotate_path: str, 
    auto_annotated_path: str, 
    mapping_file_path: str, 
    confidence: float = 0.5):   
    
    dataset_path = Path(dataset_root)
    to_annotate = to_annotate_path # dataset_path / "ToAnnotated"
    auto_annotated = auto_annotated_path #dataset_path / "AutoAnnotated"
    mapping_file = mapping_file_path # dataset_path / "master_class_mapping.xlsx"

    if not mapping_file.exists():
        raise FileNotFoundError("[ERROR] master_class_mapping.xlsx does not exist in dataset root!")

    mapping_df = pd.read_excel(mapping_file)
    species_map = {normalize_species_name(row["SpeciesName"]): int(row["ClassID"]) for _, row in mapping_df.iterrows()}
    max_class_id = max(species_map.values()) if species_map else 0

    species_folders = [f for f in to_annotate.iterdir() if f.is_dir()]
    if not species_folders:
        print("[INFO] No species to annotate.")
        return

    model = load_model(model_path)
    total_images = 0
    new_species_added = []

    for species_folder in species_folders:
        raw_name = species_folder.name
        norm_name = normalize_species_name(raw_name)

        if norm_name not in species_map:
            max_class_id += 1
            species_map[norm_name] = max_class_id
            mapping_df = pd.concat([mapping_df, pd.DataFrame([[max_class_id, raw_name]], columns=["ClassID", "SpeciesName"])])
            new_species_added.append(raw_name)

        class_id = species_map[norm_name]
        print(f"\n[•] Processing species: {raw_name} (ClassID: {class_id})")

        count = annotate_and_move_species(model, species_folder, raw_name, class_id, auto_annotated, confidence)
        total_images += count

    # Update Excel mapping
    mapping_df.to_excel(mapping_file, index=False)
    print("\n=== Summary Report ===")
    print(f"Total species processed: {len(species_folders)}")
    print(f"Total images annotated: {total_images}")
    print(f"New species added: {new_species_added if new_species_added else 'None'}")

"""


def run_annotation(
    model_path: str,
    to_annotate_dir: str = "ToAnnotate",
    auto_annotated_dir: str = "AutoAnnotated",
    mapping_csv: str = "master_class_mapping.csv",
    confidence: float = 0.5
):
    # Everything relative to where this .py file lives
    project_dir = Path(__file__).resolve().parent.parent
    to_annotate = project_dir / to_annotate_dir
    auto_annotated = project_dir / auto_annotated_dir
    mapping_file = project_dir / mapping_csv

    # Ensure output root exists
    auto_annotated.mkdir(parents=True, exist_ok=True)

    # --- Validate mapping CSV in the same folder as the code ---
    if not mapping_file.exists() or not mapping_file.is_file():
        raise FileNotFoundError(
            f"[ERROR] Mapping CSV not found beside the code: {mapping_file.resolve()}\n"
            f"Place 'master_class_mapping.csv' next to this script."
        )

    # --- Load mapping (robust to non‑UTF8) ---
    try:
        mapping_df = pd.read_csv(mapping_file)
    except UnicodeDecodeError:
        mapping_df = pd.read_csv(mapping_file, encoding="latin1")

    required_cols = {"ClassID", "SpeciesName"}
    if not required_cols.issubset(mapping_df.columns):
        raise ValueError(f"[ERROR] Mapping CSV must contain columns: {sorted(required_cols)}")

    mapping_df["ClassID"] = pd.to_numeric(mapping_df["ClassID"], errors="raise").astype(int)
    mapping_df["SpeciesName"] = mapping_df["SpeciesName"].astype(str)

    species_map = {
        normalize_species_name(row["SpeciesName"]): int(row["ClassID"])
        for _, row in mapping_df.iterrows()
    }
    max_class_id = max(species_map.values()) if species_map else 0

    # --- Check input folder next to the code ---
    if not to_annotate.exists():
        raise FileNotFoundError(f"[ERROR] ToAnnotate folder not found: {to_annotate.resolve()}")

    species_folders = [p for p in to_annotate.iterdir() if p.is_dir()]
    if not species_folders:
        print("[INFO] No species to annotate in ToAnnotate.")
        return

    # --- Load YOLO and process ---
    print(f"[INFO] Using model: {model_path}")
    model = load_model(model_path)
    total_images = 0
    new_species_added = []

    for species_folder in species_folders:
        raw_name = species_folder.name
        norm_name = normalize_species_name(raw_name)

        # Append unseen species (never delete/overwrite)
        if norm_name not in species_map:
            max_class_id += 1
            species_map[norm_name] = max_class_id
            mapping_df = pd.concat(
                [mapping_df, pd.DataFrame([[max_class_id, raw_name]], columns=["ClassID", "SpeciesName"])],
                ignore_index=True
            )
            new_species_added.append(raw_name)

        class_id = species_map[norm_name]
        print(f"\n[•] Processing species: {raw_name} (ClassID: {class_id})")

        count = annotate_and_move_species(
            model=model,
            species_path=species_folder,
            species_name=raw_name,
            class_id=class_id,
            autoannotated_root=auto_annotated,
            conf_threshold=confidence
        )
        total_images += count

    # --- Save mapping back beside the code ---
    mapping_df.to_csv(mapping_file, index=False)

    print("\n=== Summary Report ===")
    print(f"Total species processed: {len(species_folders)}")
    print(f"Total images annotated: {total_images}")
    print(f"New species added: {new_species_added if new_species_added else 'None'}")

