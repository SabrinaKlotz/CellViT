import torch
import torch.nn.functional as F
from PIL import Image
import geojson
import torchvision.transforms as T
import numpy as np
import os
import sys
import inspect
from torchvision.transforms.functional import resize, pil_to_tensor, normalize, InterpolationMode

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

def initialize_cellvit(model_name="CellViT-256-x20.pth"):
    global cellvit, cellvit_transforms, cellvit_mean, cellvit_std
    checkpoint = torch.load(os.path.join("/mnt/volume/mathias/pretrained_models/", model_name))  # Update with correct path
    config = checkpoint['config']
    model = CellViT256(model256_path=None,
                       num_nuclei_classes=config["data.num_nuclei_classes"],
                       num_tissue_classes=config["data.num_tissue_classes"],
                       regression_loss=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    inference_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    cellvit = model
    cellvit_transforms = inference_transforms
    cellvit_mean = mean
    cellvit_std = std
    cellvit.eval()

def load_tiff_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image

def process_image(image):
    print("Processing image...")
    print("Original image shape:", image.size)

    image = Image.fromarray(image)
    
    resized_image = resize(image, (1024, 1024), interpolation=InterpolationMode.BILINEAR)
    
    img_tensor = pil_to_tensor(resized_image).float().unsqueeze(0)
    
    img_norm = normalize(img_tensor, mean=cellvit_mean, std=cellvit_std)
    
    print("Processed image:")
    print("Image shape after processing:", img_norm.shape)
    
    return img_norm

def model_inference(img_norm):
    with torch.no_grad():
        predictions = cellvit(img_norm)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
        _, instance_types = cellvit.calculate_instance_map(predictions)
    return instance_types

def convert_to_geojson(instance_types):
    features = []
    for instance in instance_types:
        polygon = geojson.Polygon([instance['coordinates']])
        feature = geojson.Feature(geometry=polygon, properties={"type": instance['type']})
        features.append(feature)
    
    feature_collection = geojson.FeatureCollection(features)
    return feature_collection

def save_geojson(geojson_data, output_file):
    with open(output_file, 'w') as f:
        geojson.dump(geojson_data, f)

# Example usage:
# Initialize the model (this would be done once)
initialize_cellvit()

# Load the TIFF image
image = load_tiff_image("/mnt/volume/datasets/NCT-CRC-HE-100K/ADI/ADI-AAAMHQMK.tif")

# Preprocess the image
img_norm = process_image(image)

# Model Inference
instance_types = model_inference(img_norm)
print(instance_types)

# Convert to GeoJSON
geojson_data = convert_to_geojson(instance_types)

# Save GeoJSON
save_geojson(geojson_data, "/mnt/volume/sabrina/inference_cellvit_nct_test.geojson")
