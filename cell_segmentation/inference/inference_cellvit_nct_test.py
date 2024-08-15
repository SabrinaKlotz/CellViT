import torch
import torch.nn.functional as F
from PIL import Image
import geojson
import torchvision.transforms as T
import numpy as np

from CellViT.models.segmentation.cell_segmentation.cellvit import CellViT256

# color setup
COLOR_DICT = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
}

TYPE_NUCLEI_DICT = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

def initialize_cellvit(self, model_name="CellViT-256-x40.pth"):
    checkpoint = torch.load(self.model_dir + "CellViT/" + model_name)
    config = checkpoint['config']
    model = CellViT256(model256_path=None,
                       num_nuclei_classes=config["data.num_nuclei_classes"],
                       num_tissue_classes=config["data.num_tissue_classes"],
                       regression_loss=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    inference_transforms = T.Compose(
        [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    )
    self.cellvit = model
    self.cellvit_transforms = inference_transforms
    self.cellvit_mean = mean
    self.cellvit_std = std
    self.cellvit.eval()

def load_tiff_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image

def process_image(self, image):
    img = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()
    img = torch.nn.functional.pad(img, (0, 1024-img.shape[3], 0, 1024-img.shape[2]), value=0)
    img_norm = (img/256 - torch.tensor(self.cellvit_mean).view(3, 1, 1)/torch.tensor(self.cellvit_std).view(3, 1, 1))
    return img_norm

def model_inference(self, img_norm):
    with torch.no_grad():
        predictions = self.cellvit(img_norm)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)  # shape: (batch_size, num_nuclei_classes, H, W)
        _, instance_types = self.cellvit.calculate_instance_map(predictions)
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

# Convert to GeoJSON
geojson_data = convert_to_geojson(instance_types)

# Save GeoJSON
save_geojson(geojson_data, "/mnt/volume/sabrina/inference_cellvit_nct_test.geojson")
