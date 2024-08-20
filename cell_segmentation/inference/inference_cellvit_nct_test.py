import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys
import inspect
from torchvision.transforms.functional import resize, pil_to_tensor, normalize, InterpolationMode
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
import glob

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from models.segmentation.cell_segmentation.cellvit import CellViT256

# Global variables to store the model and transforms
cellvit = None
cellvit_transforms = None
cellvit_mean = None
cellvit_std = None

class CellViTDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.tif'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_tiff_image(image_path)
        img_norm = process_image(image)
        return img_norm, image_path

image_dir = "/mnt/volume/datasets/NCT-CRC-HE-100K/ADI"
dataset = CellViTDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

def initialize_cellvit(model_name="CellViT-256-x20.pth"):
    global cellvit, cellvit_transforms, cellvit_mean, cellvit_std
    checkpoint = torch.load(os.path.join("/mnt/volume/mathias/pretrained_models/", model_name)) 
    config = checkpoint['config']
    model = CellViT256(model256_path=None,
                       num_nuclei_classes=config["data.num_nuclei_classes"],
                       num_tissue_classes=config["data.num_tissue_classes"],
                       regression_loss=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    cellvit = model
    cellvit_mean = mean
    cellvit_std = std
    print("CellViT model loaded successfully!")
    cellvit.eval()

def load_tiff_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image

def process_image(image):
    print("Processing image...")
    print("Original image shape:", image.shape)

    image = Image.fromarray(image)
    
    resized_image = resize(image, (1024, 1024), interpolation=InterpolationMode.BILINEAR)

    img_tensor = pil_to_tensor(resized_image).float().unsqueeze(0)
    
    img_norm = normalize(img_tensor, mean=cellvit_mean, std=cellvit_std)
    
    print("Image shape after processing:", img_norm.shape)
    
    return img_norm

def model_inference(img_norm):
    with torch.no_grad():
        predictions = cellvit(img_norm)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
        _, instance_types = cellvit.calculate_instance_map(predictions, 20)
    return instance_types

def save_all_bboxes_to_json(dataloader, output_file):
    all_bbox_data = []
    
    for img_norm, image_path in dataloader:
        img_norm = img_norm.squeeze(0) 
        instance_types = model_inference(img_norm)
        
        if instance_types:  
            image_name = os.path.splitext(os.path.basename(image_path[0]))[0]

            bbox_list = []
            
            for item in instance_types:
                for key, value in item.items():
                    bbox_entry = {
                        "image_id": int(key),
                        "coordinates": value['bbox'].tolist(), 
                        "type": value['type'],                  
                        "type_prob": value['type_prob']         
                    }
                    bbox_list.append(bbox_entry)
            
            # JSON structure for each image
            json_data = {
                "img_id": int(key),               
                "height": 224,                    
                "width": 224,                     
                "file_name": image_name, 
                "bbox": bbox_list                 
            }

            all_bbox_data.append(json_data)
    
    # Save all bbox data to a single JSON file
    with open(output_file, 'w') as f:
        json.dump(all_bbox_data, f, indent=4)
    
    print(f"All bounding box data saved to {output_file}")

output_file = "/mnt/volume/sabrina/cellvit_seg/ADI.json"
save_all_bboxes_to_json(dataloader, output_file)


# Example usage:
#initialize_cellvit()

#image = load_tiff_image("/mnt/volume/datasets/NCT-CRC-HE-100K/ADI/ADI-AAAMHQMK.tif")

#img_norm = process_image(image)

#instance_types = model_inference(img_norm)
#print(instance_types)

#output_directory = "/mnt/volume/sabrina/cellvit_seg/ADI"

#save_bbox_to_json(instance_types, "/mnt/volume/datasets/NCT-CRC-HE-100K/ADI/ADI-AAAMHQMK.tif", output_directory)
