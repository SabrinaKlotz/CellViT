import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.logger import Logger

from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

from models.segmentation.cell_segmentation.cellvit import (
    CellViT256,
    CellViTSAM,
)

class InferenceCellViT:    
    
    def __init__(self, model_path: Union[Path, str], gpu: int):        
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #f"cuda:{gpu}"
        self.logger: Logger = None
        self.model = None
        self.config = None
        self.__instantiate_logger()
        self.__load_model()
        
        
    def __instantiate_logger(self) -> None:
        """
        init logger
        """
        logger = Logger(
            level="INFO",
        )
        self.logger = logger.create_logger()                        


    def __get_model(self, model_type) -> Union[CellViT256, CellViTSAM]:
        """
        get model architecture to load pretrained weights into
        """
        implemented_models = ["CellViT256", "CellViTSAM"]
        
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )

        if model_type in ["CellViT256"]:
            model = CellViT256(model256_path=None,
            num_nuclei_classes=self.config["data.num_nuclei_classes"],
            num_tissue_classes=self.config["data.num_tissue_classes"],
            regression_loss=False)
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=self.config["data.num_nuclei_classes"],
            num_tissue_classes=self.config["data.num_tissue_classes"],
            vit_structure=self.config["model.backbone"],
            regression_loss=False)
            
        return model
    
    
    def __load_model(self):
        """
        load pretrained model based on self.model_path
        """
        self.logger.info(f"Loading model: {self.model_path}")

        checkpoint = torch.load(self.model_path)
        self.config = checkpoint['config']

        # unpack checkpoint        
        self.model = self.__get_model(model_type=checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)
    
    
    def __get_dataloader(self, img_paths, batch_size):
        """
        get dataloader for dataset based on input img_paths (here we use img_paths for each nct class)
        """
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )
        dataset = PatchDataset(img_paths, inference_transforms)
        
        # copied from cell_detection.py (idk what it does)
        num_workers = int(3 / 4 * os.cpu_count())
        if num_workers is None:
            num_workers = 16
        num_workers = int(np.clip(num_workers, 1, 2 * batch_size))
        
        dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )
        return dataloader
    
    
    def inference_step(self, inference_loader, patch_class):
        """
        do inference for 1 class
        """
                
        data_l = tqdm(inference_loader, total=len(inference_loader), unit='batch')
        all_instance_data = []  # Prepare to collect all instance data
        all_file_names = []  # Collect all file names

        with torch.no_grad():
            for batch_idx, (batch, file_names) in enumerate(data_l):
                batch = batch.to(self.device)
                predictions = self.model(batch)
                predictions["nuclei_binary_map"] = F.softmax(
                    predictions["nuclei_binary_map"], dim=1
                )  # shape: (batch_size, 2, H, W)
                predictions["nuclei_type_map"] = F.softmax(
                    predictions["nuclei_type_map"], dim=1
                )  # shape: (batch_size, num_nuclei_classes, H, W)
                # get the instance types
                (
                    _,
                    instance_types,
                ) = self.model.calculate_instance_map(predictions, magnification=self.config['data.magnification'])
                
                all_instance_data.extend(instance_types) # Collect all instance_types
                all_file_names.extend(file_names) # Collect all file names
                
                data_l.set_postfix_str(f'generate bbox for {patch_class}') # displays class name in tqdm

        # After processing all batches, save the results
        return all_instance_data, all_file_names
    
    def save_bbox_to_json(self, patch_class, instance_types, file_names, output_directory):
        all_instance_data = []

        # Process each image in the batch
        for img_idx in range(len(instance_types)):
            instance_data = instance_types[img_idx]
            
            image_id = int(img_idx)  # Convert to standard Python int
            file_name = file_names[img_idx]  # Get the specific file name for this image

            # Initialize bbox list and bbox data
            bbox_list = []

            if instance_data:  # Check if instance_data is not empty
                for key, value in instance_data.items():
                    bbox = value['bbox']

                    flipped_bbox = [
                        int(bbox[0][1]), int(bbox[0][0]),  # Convert y1, x1 to int
                        int(bbox[1][1]), int(bbox[1][0])   # Convert y2, x2 to int
                    ]

                    bbox_entry = {
                        "image_id": image_id,
                        "coordinates": flipped_bbox,  # Use flipped coordinates
                        "type": int(value['type']),  # Convert to standard Python int
                        "type_prob": float(value['type_prob'])  # Ensure type_prob is a float
                    }
                    bbox_list.append(bbox_entry)

            # JSON structure for each image
            json_data = {
                "image_id": image_id,  # Use same image ID as in the instance data
                "height": 224,         # Set image height (should be batch.shape[2])
                "width": 224,          # Set image width (should be batch.shape[3])
                "file_name": file_name,  # Use the actual file name from the batch
                "bbox": bbox_list      # Bounding box data (can be empty)
            }

            all_instance_data.append(json_data)

        # After all batches are processed, save the data to a JSON file
        if all_instance_data:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            
            # Save to JSON file
            json_file = os.path.join(output_directory, f"{patch_class}_bboxes.json")
            with open(json_file, 'w') as f:
                json.dump(all_instance_data, f, indent=4)
            
            print(f"Bounding box data saved to {json_file}")


    def patch_inference(self, root_dir, file_extension, batch_size):
        """
        runner for patch inference
        takes root_dir and run inference for each class
        
        """
        # gets all files with file_extension from root_dir
        patch_files = sorted(Path(root_dir).glob(f"**/*{file_extension}")) # maybe needs filtering
        
        # divide file_paths list into dictionary: keys=classes : value=list of image_paths        
        class_dict = {}
        for file_path in patch_files:
            class_name = file_path.parts[-2]

            if class_name not in class_dict:
                class_dict[class_name] = []  

            class_dict[class_name].append(file_path)
            
        for patch_class, img_paths in class_dict.items():
            
            ############# custom testing only run for ADI, remove for final version #####################
            #if patch_class != "BACK":
                #continue
            
            inference_loader = self.__get_dataloader(img_paths=img_paths, batch_size=batch_size)
            instance_types, file_names = self.inference_step(inference_loader=inference_loader, patch_class=patch_class)
            
            self.save_bbox_to_json(patch_class=patch_class, instance_types=instance_types, file_names=file_names, output_directory=f"/mnt/volume/sabrina/cellvit_seg/{patch_class}")
        

class PatchDataset(Dataset):
    """
    Dataset for patches e.g., NCT-CRC-100k
    """
    def __init__(self, image_paths, transform=None):
        super(PatchDataset, self).__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, img_path.name  # Return both the image tensor and the file name


if __name__ == "__main__":
    img_root = "/mnt/volume/datasets/NCT-CRC-HE-100K/"
    file_extension = ".tif"
    model_path = "/mnt/volume/mathias/pretrained_models/CellViT-256-x20.pth"
    gpu = 0
    batch_size = 128 # 128 highest maybe try 256
    
    cell_detection = InferenceCellViT(model_path=model_path, gpu=gpu)
    cell_detection.patch_inference(root_dir=img_root, file_extension=file_extension, batch_size=batch_size)
