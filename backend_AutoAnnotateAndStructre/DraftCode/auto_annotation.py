"""
Author: Andrew Shepley
Contact: asheple2@une.edu.au
Source: U-Infuse
Purpose: Functions for inference and visualisation
Methods
a) __init__
b) prettify
c) get_labels_boxes
d) create_annotations_dir
e) write_xml_file
f) get_model_instance_segmentation
g) make_prediction
h) load_inference_image
i) process_predictions


Upadate by: Redwan Hossain. 
Linked In Contatc: www.linkedin.com/in/hossainredu
Purpose: Use pretarined Yolov12 model for auto annotation. 
Changes made: 
- Load model in cuda (gpu) if available. Prevously onlu cpu was ussed. 
- Add write_txt_annotation function for Yolo model tranning annotation. 
- added Load_model_yolo to load pre trained yolo model. ( previous load model  
"""

from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.dom import minidom
import os
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, datasets, models
import os
from PIL import Image
import  numpy as np
from torch.autograd import Variable

from ultralytics import YOLO
import torch
from PIL import Image


class Auto_Annotator():

    #class variables
    dataset_dir = ""
    image_name = ""
    image_dims = []
    annotation_path = "" # "./annotations/"
    confidence_threshold = 0.5
    object_detection_data = []

    def __init__(self, dataset_dir, annotation_path, image_name, object_detection_data, image_dims, confidence_threshold):
        """
        Parameters:
           dataset_dir: dir containing image to annotate
           image_name: name of image
           object_detection_data: annotation info; class label, scores, bounding box (list)
           image_dims: width, height, channels (list)
           confidence_threshold: user preset conf. threshold
        """
        self.dataset_dir = dataset_dir
        self.annotation_path = annotation_path
        self.image_name = image_name
        self.object_detection_data = object_detection_data
        self.image_dims = image_dims
        self.confidence_threshold

    def get_labels_boxes(self):
        """
        Function: returns list of class labels with associated bounding boxes
        """
        boxes = []
        labels = []
        for each in self.object_detection_data:
            labels.append(each[0])
            boxes.append(each[2].tolist())
        return boxes, labels

    def create_annotations_dir(self, folder):
        """
        Parameters: 
          folder: name of folder containing images
        Function: if the annotations_dir doesn't exist, create it
        """
        annotations_dir = folder #os.path.join('./annotations/',folder)
        if os.path.exists(annotations_dir) is False:
            os.mkdir(annotations_dir)
        return

    def prettify(self, elem):
        """
        Function: Return a pretty-printed XML string
        """
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def write_xml_file(self, annotations_root_dir="."):
        """
        Function: writes xml annotation file for image 
        """
        boxes, labels = self.get_labels_boxes()
        folder_name = self.dataset_dir.split("/")[2]
        full_folder_name = annotations_root_dir + "/" + folder_name
        
        self.create_annotations_dir(full_folder_name) #folder_name)

        root = Element( 'annotation' )
        folder = SubElement( root, 'folder' )
        folder.text = folder_name
        filename = SubElement(root, 'filename')
        filename.text = self.image_name
        path = SubElement(root, 'path')
        path.text = self.dataset_dir
        source = SubElement(root, 'source')
        database = SubElement(source, 'database')
        database.text = 'U-Infuse FlickR'
    
        size = SubElement(root, 'size')
        width = SubElement(size, 'width')
        width.text = str(self.image_dims[1])
        height = SubElement(size, 'height')
        height.text = str(self.image_dims[0])
        depth = SubElement(size, 'depth')
        depth.text = str(self.image_dims[2])    
        segmented = SubElement(root, 'segmented')
        segmented.text = str(0)

        for b, each_label in zip(boxes, labels):
            object_ = SubElement(root, 'object')
            name_ = SubElement(object_, 'name')
            name_.text = each_label
            pose = SubElement(object_, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(object_, 'truncated')
            truncated.text = '0'
            difficult = SubElement(object_, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(object_, 'bndbox')
            xmin= SubElement(bndbox, 'xmin')
            xmin.text = str(b[0])
            ymin= SubElement(bndbox, 'ymin')
            ymin.text = str(b[1])
            xmax= SubElement(bndbox, 'xmax')
            xmax.text = str(b[2])
            ymax= SubElement(bndbox, 'ymax')
            ymax.text = str(b[3])

        output_file = open(self.annotation_path+"/"+folder_name+"/"+self.image_name[:-4]+'.xml', 'w' )
        output_file.write(self.prettify(root))
    
        output_file.close()
        return 

    def write_txt_annotation(self):
        boxes, labels = self.get_labels_boxes()
        class_id = 0
        annotation_lines = []
        for b in boxes:
            xmin, ymin, xmax, ymax = b
            img_w, img_h = self.image_dims[1], self.image_dims[0]
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w = (xmax - xmin) / img_w
            box_h = (ymax - ymin) / img_h
            annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        output_path = os.path.join(self.annotation_path, self.image_name.rsplit('.', 1)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write("\n".join(annotation_lines))
        return

def get_model_instance_segmentation(num_classes):
  
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def load_inference_image_fasterRcnn(model,img_path):
    image = Image.open(img_path).convert("RGB")
    loader = transforms.Compose([transforms.ToTensor()])
    im = loader(image).float()
    im = Variable(im, requires_grad=True)
    im = im.unsqueeze(0)
    #im=im.cuda()

    with torch.no_grad(): 
        model.eval()
        preds = model(im)
        return image,preds[0]

        
def load_inference_image_yolo(model, img_path):
    image = Image.open(img_path).convert("RGB")
    results = model.predict(source=img_path, save=False, verbose=False)
    return image, results[0]
        

def load_model_fasterRcnn(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print("[✓] CUDA is available. Loading model on GPU.")
    else:
        print("[!] CUDA is NOT available. Loading model on CPU. This may be slower.")
    
    model = get_model_instance_segmentation(4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def load_model_yolo(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("[✓] CUDA is available. Loading YOLOv12m model on GPU.")
    else:
        print("[!] CUDA is NOT available. Loading on CPU — slower performance expected.")

    model = YOLO(model_path)
    model.to(device)
    return model


def process_predictions_fasterRcnn(preds,confidence_threshold,class_mapping):
    object_data =[]
    for i in range(len(preds['boxes'])):
        score=preds['scores'][i]
        if score > confidence_threshold:
            per_ant=[]
            label=int(preds['labels'][i])
            class_label=class_mapping[label]
            box=preds['boxes'][i].cpu().numpy().astype(int)#.tolist()
            per_ant.append(class_label)
            per_ant.append(score)
            per_ant.append(box)
            object_data.append(per_ant)
    return object_data

def process_predictions_yolo(preds, confidence_threshold, class_label="ant"):
    object_data = []

    if preds.boxes is None or len(preds.boxes) == 0:
        return object_data

    for box in preds.boxes:
        score = float(box.conf)
        if score >= confidence_threshold:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [xmin, ymin, xmax, ymax]
            object_data.append([class_label, score, xyxy])

    return object_data    