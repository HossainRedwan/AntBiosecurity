"""
Authour: Md Redwan Hossain
Date: 02/08/2025

Objective: Copies image data from a user-specified dataset folder to a "ToAnnotate" folder located in the same directory as the script. Organsie dataset for auto annotationa and Yolo model traning. 

Features:
    - Recursively searches all speceis subfolders for images in a user specified datset folder.
    - Creates or looks for a ToAnnotate folder next to your notebook/script.
    - Creates ot look for a species folders with a nested <Species>_Images folder.
    - Renames files with timestamp for uniqueness.
    - Normalizes folder matching (ignores case, spaces, underscores, and dashes).
    - Prints clear progress and summary report.
"""



import os
import shutil
from pathlib import Path
import hashlib

class ImageCopier:
    VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    def __init__(self, dataset_folder: Path, output_folder: Path = None):
        """
        Initialize ImageCopier.
        dataset_folder: Path to dataset containing species folders.
        output_folder: Path where ToAnnotate folder will be created (defaults to current working dir).
        """
        self.dataset_folder = Path(dataset_folder)
        self.output_folder = Path(output_folder) if output_folder else Path.cwd()
        self.to_annotate_dir = self.output_folder / "ToAnnotate"
        self.to_annotate_dir.mkdir(exist_ok=True)

        # Summary counters
        self.species_processed = 0
        self.total_images_copied = 0
        self.created_species_folders = []

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize folder names for comparison by ignoring spaces, underscores, and dashes, and lowercasing."""
        return name.replace(" ", "").replace("-", "").replace("_", "").lower()

    @classmethod
    def is_image_file(cls, file_path: Path) -> bool:
        """Check if the file is an image based on extension."""
        return file_path.suffix.lower() in cls.VALID_EXTENSIONS

    @staticmethod
    def short_hash(path: Path, length=6) -> str:
        """Generate a short hash for the file based on full source path."""
        return hashlib.md5(str(path).encode()).hexdigest()[:length]

    @classmethod
    def get_hashed_filename(cls, file_path: Path) -> str:
        """Return filename with short hash appended before extension."""
        hash_suffix = cls.short_hash(file_path)
        return f"{file_path.stem}_{hash_suffix}{file_path.suffix.lower()}"

    def copy_images(self):
        """Copy images to ToAnnotate folder maintaining species-based folder structure."""
        # Map normalized names to existing folders in ToAnnotate
        existing_folders = {self.normalize_name(f.name): f 
                            for f in self.to_annotate_dir.iterdir() if f.is_dir()}

        for species_folder in self.dataset_folder.iterdir():
            if not species_folder.is_dir():
                continue

            species_name = species_folder.name
            species_norm = self.normalize_name(species_name)

            # Find or create species folder in ToAnnotate
            if species_norm in existing_folders:
                target_species_folder = existing_folders[species_norm]
            else:
                target_species_folder = self.to_annotate_dir / species_name
                target_species_folder.mkdir(exist_ok=True)
                self.created_species_folders.append(species_name)
                existing_folders[species_norm] = target_species_folder

            # Create Images folder
            images_folder = target_species_folder / f"{species_folder.name}_Images"
            images_folder.mkdir(exist_ok=True)

            print(f"[•] Processing species: {species_folder.name}")
            self.species_processed += 1

            # Recursively copy images with hash in filename
            for img_path in species_folder.rglob("*"):
                if img_path.is_file() and self.is_image_file(img_path):
                    new_filename = self.get_hashed_filename(img_path)
                    dest_path = images_folder / new_filename
                    shutil.copy2(img_path, dest_path)
                    self.total_images_copied += 1

        # Summary
        self._print_summary()

    def _print_summary(self):
        print("\n========== Summary Report ==========")
        print(f"Total species processed: {self.species_processed}")
        print(f"Total images copied: {self.total_images_copied}")
        print(f"New added species folders: {self.created_species_folders}")
        print("===================================")
