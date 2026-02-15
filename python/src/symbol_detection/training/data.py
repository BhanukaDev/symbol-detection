import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        coco_json_path: str | Path,
        images_dir: str | Path,
        transform=None,
    ):
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        with open(self.coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.image_ids[idx]
        img_meta = self.images[img_id]
        
        img_path = self.images_dir / img_meta['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.annotations_by_image.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            areas = np.zeros((0,), dtype=np.float32)
            iscrowd = np.zeros((0,), dtype=np.uint8)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            areas = np.array(areas, dtype=np.float32)
            iscrowd = np.array(iscrowd, dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels),
            'area': torch.from_numpy(areas).float(),
            'iscrowd': torch.from_numpy(iscrowd),
            'image_id': torch.tensor(img_id),
        }

        return image, target


def collate_fn(batch):
    """Custom collate function for variable-sized images and bounding boxes."""
    images, targets = zip(*batch)
    # Keep images as list (not stacked) - not all same size
    return list(images), list(targets)
