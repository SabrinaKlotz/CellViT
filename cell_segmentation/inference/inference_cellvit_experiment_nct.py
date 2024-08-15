import torch
import torch.nn.functional as F
from PIL import Image
import geojson
import torchvision.transforms as T
import numpy as np
import pandas as pd
import logging
import uuid
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon 
from rtree import index
from rtree.index import STRtree
from collections import deque

from models.segmentation.cell_segmentation.cellvit import CellViT256

from utils.logger import Logger
from utils.tools import unflatten_dict

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

def convert_geojson(cell_list: list[dict], polygons: bool = False) -> list[dict]:
    """Convert a list of cells to a geojson object

    Either a segmentation object (polygon) or detection points are converted

    Args:
        cell_list (list[dict]): Cell list with dict entry for each cell.
            Required keys for detection:
                * type
                * centroid
            Required keys for segmentation:
                * type
                * contour
        polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

    Returns:
        list[dict]: Geojson like list
    """
    if polygons:
        cell_segmentation_df = pd.DataFrame(cell_list)
        detected_types = sorted(cell_segmentation_df.type.unique())
        geojson_placeholder = []
        for cell_type in detected_types:
            cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
            contours = cells["contour"].to_list()
            final_c = []
            for c in contours:
                c.append(c[0])
                final_c.append([c])

            cell_geojson_object = get_template_segmentation()
            cell_geojson_object["id"] = str(uuid.uuid4())
            cell_geojson_object["geometry"]["coordinates"] = final_c
            cell_geojson_object["properties"]["classification"]["name"] = TYPE_NUCLEI_DICT[cell_type]
            cell_geojson_object["properties"]["classification"]["color"] = COLOR_DICT[cell_type]
            geojson_placeholder.append(cell_geojson_object)
    else:
        cell_detection_df = pd.DataFrame(cell_list)
        detected_types = sorted(cell_detection_df.type.unique())
        geojson_placeholder = []
        for cell_type in detected_types:
            cells = cell_detection_df[cell_detection_df["type"] == cell_type]
            centroids = cells["centroid"].to_list()
            cell_geojson_object = get_template_point()
            cell_geojson_object["id"] = str(uuid.uuid4())
            cell_geojson_object["geometry"]["coordinates"] = centroids
            cell_geojson_object["properties"]["classification"]["name"] = TYPE_NUCLEI_DICT[cell_type]
            cell_geojson_object["properties"]["classification"]["color"] = COLOR_DICT[cell_type]
            geojson_placeholder.append(cell_geojson_object)
    return geojson_placeholder

def get_template_segmentation():
    """Returns a template for a segmentation geojson object"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": []
        },
        "properties": {
            "classification": {
                "name": "",
                "color": ""
            }
        },
        "id": ""
    }

def get_template_point():
    """Returns a template for a point geojson object"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": []
        },
        "properties": {
            "classification": {
                "name": "",
                "color": ""
            }
        },
        "id": ""
    }

def convert_coordinates(row: pd.Series) -> pd.Series:
    """Convert a row from x,y type to one string representation of the patch position for fast querying
    Repr: x_y

    Args:
        row (pd.Series): Row to be processed

    Returns:
        pd.Series: Processed Row
    """
    x, y = row["patch_coordinates"]
    row["patch_row"] = x
    row["patch_col"] = y
    row["patch_coordinates"] = f"{x}_{y}"
    return row

def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> list[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        list[int]: List with 4 integers for each position
    """
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position

def get_cell_position_margin(bbox: np.ndarray, patch_size: int = 1024, margin: int = 64) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            if bbox[0, 1] < margin:
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                cell_status = 3
            else:
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            if bbox[1, 0] > patch_size - margin:
                cell_status = 5
            else:
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            if bbox[0, 1] < margin:
                cell_status = 7
            else:
                cell_status = 6
        elif bbox[0, 1] < margin:
            cell_status = 8
    else:
        cell_status = 0

    return cell_status

def get_edge_patch(position, row, col):
    if position == [1, 0, 0, 0]:
        return [[row - 1, col]]
    if position == [1, 1, 0, 0]:
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 1, 0, 0]:
        return [[row, col + 1]]
    if position == [0, 1, 1, 0]:
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 0, 1, 0]:
        return [[row + 1, col]]
    if position == [0, 0, 1, 1]:
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [0, 0, 0, 1]:
        return [[row, col - 1]]
    if position == [1, 0, 0, 1]:
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]

class CellPostProcessor:
    def __init__(self, cell_list: list[dict], logger: logging.Logger) -> None:
        """Post-Processing a list of cells from one WSI

        Args:
            cell_list (list[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger
        self.logger.info("Initializing Cell-Postprocessor")
        self.cell_df = pd.DataFrame(cell_list)
        self.cell_df = self.cell_df.apply(convert_coordinates, axis=1)

        self.mid_cells = self.cell_df[
            self.cell_df["cell_status"] == 0
        ]
        self.cell_df_margin = self.cell_df[
            self.cell_df["cell_status"] != 0
        ]

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]
        ).sort_index()
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Clean margin and edge cells by removing overlaps.

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        margin_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 0
        ]
        edge_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 1
        ]
        existing_patches = list(
            set(self.cell_df_margin["patch_coordinates"].to_list())
        )

        edge_cells_unique = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )

        for idx, cell_info in edge_cells.iterrows():
            edge_information = dict(cell_info["edge_information"])
            edge_patch = edge_information["edge_patches"][0]
            edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
            if edge_patch not in existing_patches:
                edge_cells_unique.loc[idx, :] = cell_info

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])

        return cleaned_edge_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from the provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame to be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = cleaned_edge_cells

        for iteration in range(20):
            poly_list = []
            for idx, cell_info in merged_cells.iterrows():
                poly = Polygon(cell_info["contour"])
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    if isinstance(multi, MultiPolygon):
                        if len(multi) > 1:
                            poly_idx = np.argmax([p.area for p in multi])
                            poly = multi[poly_idx]
                        else:
                            poly = multi[0]
                    poly = Polygon(poly)
                poly.uid = idx
                poly_list.append(poly)

            tree = STRtree(poly_list)
            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0

            for query_poly in poly_list:
                if query_poly.uid not in iterated_cells:
                    intersected_polygons = tree.query(query_poly)
                    if len(intersected_polygons) > 1:
                        submergers = []
                        for inter_poly in intersected_polygons:
                            if inter_poly.uid != query_poly.uid and inter_poly.uid not in iterated_cells:
                                if query_poly.intersection(inter_poly).area / query_poly.area > 0.01 or \
                                   query_poly.intersection(inter_poly).area / inter_poly.area > 0.01:
                                    overlaps += 1
                                    submergers.append(inter_poly)
                                    iterated_cells.add(inter_poly.uid)
                        if len(submergers) == 0:
                            merged_idx.append(query_poly.uid)
                        else:
                            selected_poly_index = np.argmax(np.array([p.area for p in submergers]))
                            selected_poly_uid = submergers[selected_poly_index].uid
                            merged_idx.append(selected_poly_uid)
                    else:
                        merged_idx.append(query_poly.uid)
                    iterated_cells.add(query_poly.uid)

            self.logger.info(f"Iteration {iteration}: Found overlap of # cells: {overlaps}")
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
            elif iteration == 20:
                self.logger.info(
                    f"Not all doubled cells removed, still {overlaps} to remove. For performance issues, we stop iterations now."
                )
            merged_cells = cleaned_edge_cells.loc[cleaned_edge_cells.index.isin(merged_idx)].sort_index()

        return merged_cells.sort_index()

    def convert_to_geojson(self, polygons: bool = False) -> list[dict]:
        """Convert the post-processed cells to GeoJSON format

        Args:
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            list[dict]: GeoJSON-like list
        """
        self.logger.info("Converting post-processed cells to GeoJSON")
        cell_list = self.post_process_cells().to_dict(orient='records')
        geojson_data = convert_geojson(cell_list, polygons=polygons)
        return geojson_data

def save_geojson(geojson_data, output_file):
    with open(output_file, 'w') as f:
        geojson.dump(geojson_data, f)

# Example usage:
# Initialize the model (this would be done once)
initialize_cellvit()

# Load the TIFF image
image = load_tiff_image("your_image_path.tiff")

# Preprocess the image
img_norm = process_image(image)

# Model Inference
instance_types = model_inference(img_norm)

# Create an instance of CellProcessor
processor = CellPostProcessor(instance_types, logger)

# Post-process cells
post_processed_cells = processor.post_process_cells()

# Convert to GeoJSON (either polygons or detection points)
geojson_data = processor.convert_to_geojson(polygons=True)

# Save or use geojson_data as needed
save_geojson(geojson_data, "output_file.geojson")