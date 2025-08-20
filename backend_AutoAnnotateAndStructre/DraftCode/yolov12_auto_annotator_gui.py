import tkinter as tk
from tkinter import filedialog, messagebox
from backend_auto_annotator_yolov12 import run_annotation
import subprocess
import os

def browse_dataset():
    path = filedialog.askdirectory()
    if path:
        dataset_entry.delete(0, tk.END)
        dataset_entry.insert(0, path)

def browse_model():
    path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
    if path:
        model_entry.delete(0, tk.END)
        model_entry.insert(0, path)

def run_annotator():
    dataset = dataset_entry.get().strip()
    model = model_entry.get().strip()
    try:
        confidence = float(confidence_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Confidence threshold must be a number.")
        return

    if not dataset or not model:
        messagebox.showerror("Error", "Please provide both dataset and model path.")
        return

    try:
        run_annotation(dataset_root=dataset, model_path=model, confidence=confidence)
        messagebox.showinfo("Done", "Annotation completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run annotator:\n{e}")

def launch_labelimg():
    try:
        species_folders = [f for f in os.listdir(dataset_entry.get()) if os.path.isdir(os.path.join(dataset_entry.get(), f))]
        if not species_folders:
            messagebox.showerror("Error", "No species folders found in dataset.")
            return
        first_annot_folder = os.path.join(dataset_entry.get(), species_folders[0], "annotations")
        subprocess.run(["labelImg", first_annot_folder])
    except Exception as e:
        messagebox.showerror("Error", f"Could not launch labelImg:\n{e}")

# GUI setup
root = tk.Tk()
root.title("YOLOv12 Ant Auto Annotator")

tk.Label(root, text="Dataset Folder:").grid(row=0, column=0, sticky="e")
dataset_entry = tk.Entry(root, width=50)
dataset_entry.grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_dataset).grid(row=0, column=2)

tk.Label(root, text="Model File (.pt):").grid(row=1, column=0, sticky="e")
model_entry = tk.Entry(root, width=50)
model_entry.grid(row=1, column=1)
tk.Button(root, text="Browse", command=browse_model).grid(row=1, column=2)

tk.Label(root, text="Confidence Threshold:").grid(row=2, column=0, sticky="e")
confidence_entry = tk.Entry(root, width=10)
confidence_entry.insert(0, "0.5")
confidence_entry.grid(row=2, column=1, sticky="w")

tk.Button(root, text="Run Auto Annotation", command=run_annotator, bg="green", fg="white").grid(row=3, column=1, pady=10)
tk.Button(root, text="Launch labelImg", command=launch_labelimg).grid(row=4, column=1, pady=5)

root.mainloop()
