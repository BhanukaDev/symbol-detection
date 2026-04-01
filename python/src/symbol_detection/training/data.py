import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainingAugmentation:
    """Augmentations applied during training for better generalization."""

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        """
        Apply random augmentations to image and bounding boxes.

        Args:
            image: HWC uint8 RGB numpy array.
            boxes: Nx4 float32 array in [x1, y1, x2, y2] format.
            labels: N int64 array.

        Returns:
            Augmented (image, boxes, labels).
        """
        h, w = image.shape[:2]

        # Horizontal flip (50%)
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            if len(boxes) > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1

        # Random brightness jitter
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Random contrast jitter
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            mean = image.mean()
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random scale jitter (resize image by 0.8-1.2x, adjust boxes)
        if random.random() < 0.4:
            scale = random.uniform(0.8, 1.2)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if len(boxes) > 0:
                boxes *= scale

        return image, boxes, labels


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        coco_json_path: str | Path,
        images_dir: str | Path,
        transform=None,
        max_size: int = 640,
        augment: bool = False,
    ):
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_size = max_size
        self.augmentation = TrainingAugmentation() if augment else None
        
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
        
        # Important: Sort categories to ensure consistent indexing across runs
        sorted_categories = sorted(self.coco_data['categories'], key=lambda x: x['id'])
        self.categories = {cat['id']: cat['name'] for cat in sorted_categories}
        # Important: FasterRCNN expects 0 to be background.
        # So we map our categories to 1-based indices.
        self.cat_id_to_idx = {cat['id']: idx + 1 for idx, cat in enumerate(sorted_categories)}

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

        # Apply training augmentations (before resize)
        if self.augmentation is not None and len(boxes) > 0:
            image, boxes, labels = self.augmentation(image, boxes, labels)
            # Recompute areas after augmentation
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        if self.transform:
            image = self.transform(image)
        else:
            # Resize if image exceeds max_size (preserving aspect ratio)
            h, w = image.shape[:2]
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= scale
                    boxes[:, [1, 3]] *= scale
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
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
