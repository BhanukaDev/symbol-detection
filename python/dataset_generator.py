"""
Dataset Generator for Floor Plan Symbol Detection

Generates synthetic floor plan images with electrical symbols and
exports annotations in COCO format.
"""

import cv2
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from floor_grid import generate_building_with_symbols


class COCODatasetGenerator:
    """Generates floor plan dataset with COCO format annotations."""

    def __init__(self, output_dir: str = "dataset", symbols_dir: str = "data/electrical-symbols"):
        """
        Initialize the dataset generator.

        Args:
            output_dir: Directory where images and annotations will be saved.
            symbols_dir: Directory containing electrical symbol images.
        """
        self.output_dir = Path(output_dir)
        self.symbols_dir = symbols_dir
        self.images_dir = self.output_dir / "images"
        self.annotations_file = self.output_dir / "annotations.json"
        
        # COCO format data structures
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Track category IDs
        self.category_name_to_id: Dict[str, int] = {}
        self.next_category_id = 1
        self.next_annotation_id = 1

    def _get_or_create_category_id(self, category_name: str) -> int:
        """
        Get existing category ID or create a new one.

        Args:
            category_name: Name of the symbol category.

        Returns:
            Category ID.
        """
        if category_name not in self.category_name_to_id:
            category_id = self.next_category_id
            self.category_name_to_id[category_name] = category_id
            self.coco_data["categories"].append({
                "id": category_id,
                "name": category_name
            })
            self.next_category_id += 1
            return category_id
        return self.category_name_to_id[category_name]

    def _convert_annotations_to_coco(
        self,
        annotations: List[Dict],
        image_id: int
    ) -> List[Dict]:
        """
        Convert symbol annotations to COCO format.

        Args:
            annotations: List of symbol annotations from generate_building_with_symbols.
            image_id: ID of the image these annotations belong to.

        Returns:
            List of COCO format annotation dictionaries.
        """
        coco_annotations = []

        for ann in annotations:
            category_id = self._get_or_create_category_id(ann["class_name"])
            
            # COCO bbox format: [x, y, width, height] where x,y is top-left corner
            bbox = [
                float(ann["x"]),
                float(ann["y"]),
                float(ann["width"]),
                float(ann["height"])
            ]
            
            area = float(ann["width"] * ann["height"])

            coco_ann = {
                "id": self.next_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            }
            
            coco_annotations.append(coco_ann)
            self.next_annotation_id += 1

        return coco_annotations

    def generate_dataset(
        self,
        num_images: int = 10,
        rows: int = 80,
        cols: int = 120,
        cell_size: int = 20,
        door_size: int = 2,
        min_room_width: int = 8,
        min_building_area_ratio: float = 0.6,
        symbols_per_room: Tuple[int, int] = (1, 4),
        scale_range: Tuple[float, float] = (0.8, 1.5),
        rotation_range: Tuple[float, float] = (0.0, 360.0),
        symbol_classes: Optional[List[str]] = None,
        show_labels: bool = False,
    ) -> Dict:
        """
        Generate a complete dataset with COCO format annotations.

        Args:
            num_images: Number of images to generate.
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            cell_size: Size of each cell in pixels.
            door_size: Size of doors in cells.
            min_room_width: Minimum width of a room in cells.
            min_building_area_ratio: Target ratio of building area to total grid area.
            symbols_per_room: (min, max) number of symbols per room.
            scale_range: (min, max) scale factor range for symbols.
            rotation_range: (min, max) rotation angle range in degrees.
            symbol_classes: List of symbol classes to use, or None for all.
            show_labels: Whether to show room labels on the images.

        Returns:
            Dictionary with COCO format data.
        """
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {num_images} dataset images...")

        for i in range(num_images):
            image_id = i + 1
            
            # Generate building with symbols
            floor_plan_img, annotations, grid, rooms = generate_building_with_symbols(
                rows=rows,
                cols=cols,
                cell_size=cell_size,
                door_size=door_size,
                min_room_width=min_room_width,
                min_building_area_ratio=min_building_area_ratio,
                symbols_dir=self.symbols_dir,
                symbols_per_room=symbols_per_room,
                scale_range=scale_range,
                rotation_range=rotation_range,
                symbol_classes=symbol_classes,
                show_labels=show_labels,
            )

            # Get image dimensions
            height, width = floor_plan_img.shape[:2]

            # Save the image
            filename = f"floor_plan_{i:04d}.png"
            output_path = self.images_dir / filename
            cv2.imwrite(str(output_path), floor_plan_img)

            # Add image info to COCO data
            self.coco_data["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": int(width),
                "height": int(height)
            })

            # Convert and add annotations
            coco_annotations = self._convert_annotations_to_coco(annotations, image_id)
            self.coco_data["annotations"].extend(coco_annotations)

            print(
                f"[{i+1}/{num_images}] Generated {filename} - "
                f"{len(rooms)} rooms, {len(annotations)} symbols"
            )

        print(f"\nDataset generation complete!")
        print(f"  - Images: {num_images} saved to '{self.images_dir}/'")
        print(f"  - Categories: {len(self.coco_data['categories'])}")
        print(f"  - Total annotations: {len(self.coco_data['annotations'])}")
        
        return self.coco_data

    def save_annotations(self):
        """Save COCO format annotations to JSON file."""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"Annotations saved to '{self.annotations_file}'")

    def get_coco_data(self) -> Dict:
        """
        Get the COCO format dataset.

        Returns:
            Dictionary containing images, annotations, and categories in COCO format.
        """
        return self.coco_data


def generate_coco_dataset(
    output_dir: str = "dataset",
    symbols_dir: str = "data/electrical-symbols",
    num_images: int = 10,
    **kwargs
) -> Dict:
    """
    Convenience function to generate a COCO format dataset.

    Args:
        output_dir: Directory where images and annotations will be saved.
        symbols_dir: Directory containing electrical symbol images.
        num_images: Number of images to generate.
        **kwargs: Additional parameters passed to generate_dataset().

    Returns:
        Dictionary with COCO format data.
    """
    generator = COCODatasetGenerator(output_dir=output_dir, symbols_dir=symbols_dir)
    coco_data = generator.generate_dataset(num_images=num_images, **kwargs)
    generator.save_annotations()
    return coco_data
