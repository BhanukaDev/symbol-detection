"""
Dataset Generator for Floor Plan Symbol Detection

Generates synthetic floor plan images with electrical symbols and
exports annotations in COCO format.
"""

import cv2
import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from floor_grid import generate_building_with_symbols

try:
    from effects import water_wave_distortion, twirl_distortion
except ImportError:
    water_wave_distortion = None
    twirl_distortion = None


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

    def _apply_symbol_effects(self, img: np.ndarray, effect_type: str = "random") -> np.ndarray:
        """
        Apply distortion effects to a symbol image.

        Args:
            img: Input image (symbol).
            effect_type: Type of effect ("water_wave", "twirl", or "random").

        Returns:
            Processed image with effects applied.
        """
        if effect_type == "random":
            effect_type = random.choice(["water_wave", "twirl", "none"])
        
        try:
            if effect_type == "water_wave" and water_wave_distortion:
                amplitude = int(random.uniform(0.5, 2.0))
                frequency = random.uniform(0.01, 0.03)
                return water_wave_distortion(img, amplitude=amplitude, frequency=frequency)
            
            elif effect_type == "twirl" and twirl_distortion:
                angle = random.uniform(0.2, 0.6)
                radius = int(min(img.shape[:2]) // 2)
                return twirl_distortion(img, angle=angle, radius=radius)
        except Exception as e:
            print(f"Warning: Effect application failed: {e}")
        
        return img

    def _apply_image_effects(self, img: np.ndarray) -> np.ndarray:
        """
        Apply effects to the entire image (noise, color changes, brightness).

        Args:
            img: Input image.

        Returns:
            Processed image with effects.
        """
        result: np.ndarray = img.copy()
        
        # Add Gaussian noise
        if random.random() < 0.6:
            noise_intensity = random.uniform(0.001, 0.01)
            noise = np.random.normal(0, 255 * noise_intensity, result.shape)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.85, 1.15)
            result = np.clip(result.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = result.mean()
            result = np.clip((result.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Random color shift (hue adjustment for color images)
        if len(result.shape) == 3 and result.shape[2] == 3 and random.random() < 0.3:
            try:
                # Ensure result is uint8 and contiguous
                result = np.ascontiguousarray(result, dtype=np.uint8)
                # Convert BGR to HSV
                hsv: np.ndarray = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                # Convert to float for manipulation
                hsv_float = hsv.astype(np.float32)
                # Adjust hue channel
                hsv_float[:, :, 0] = np.mod(hsv_float[:, :, 0] + random.uniform(-10, 10), 180)
                # Convert back to uint8
                hsv_uint8 = np.ascontiguousarray(np.uint8(np.clip(hsv_float, 0, 255)))
                # Convert HSV back to BGR
                result = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)
            except Exception as e:
                print(f"Warning: Color shift failed: {e}")
        
        return result

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
        rows: Optional[Tuple[int, int]] = None,
        cols: Optional[Tuple[int, int]] = None,
        cell_size: Union[Tuple[int, int], int] = 20,
        door_size: int = 2,
        min_room_width: int = 8,
        min_building_area_ratio: float = 0.6,
        symbols_per_room: Tuple[int, int] = (1, 4),
        scale_range: Tuple[float, float] = (0.8, 1.5),
        rotation_range: Tuple[float, float] = (0.0, 360.0),
        symbol_classes: Optional[List[str]] = None,
        show_labels: bool = False,
        apply_symbol_effects: bool = True,
        apply_image_effects: bool = True,
    ) -> Dict:
        """
        Generate a complete dataset with COCO format annotations.

        Args:
            num_images: Number of images to generate.
            rows: Number of rows or (min, max) range for random variation.
            cols: Number of columns or (min, max) range for random variation.
            cell_size: Size of each cell in pixels or (min, max) range for variation.
            door_size: Size of doors in cells.
            min_room_width: Minimum width of a room in cells.
            min_building_area_ratio: Target ratio of building area to total grid area.
            symbols_per_room: (min, max) number of symbols per room.
            scale_range: (min, max) scale factor range for symbols.
            rotation_range: (min, max) rotation angle range in degrees.
            symbol_classes: List of symbol classes to use, or None for all.
            show_labels: Whether to show room labels on the images.
            apply_symbol_effects: Whether to apply distortion effects to symbols.
            apply_image_effects: Whether to apply noise/color effects to images.

        Returns:
            Dictionary with COCO format data.
        """
        # Set default ranges if not provided
        if rows is None:
            rows = (60, 100)
        if cols is None:
            cols = (90, 150)
        
        # Convert single values to ranges
        if isinstance(rows, int):
            rows = (rows, rows)
        if isinstance(cols, int):
            cols = (cols, cols)
        if isinstance(cell_size, int):
            cell_size = (cell_size, cell_size)
        
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {num_images} dataset images with varied dimensions...")
        print(f"  - Rows range: {rows[0]} to {rows[1]}")
        print(f"  - Cols range: {cols[0]} to {cols[1]}")
        print(f"  - Cell size range: {cell_size[0]} to {cell_size[1]} pixels")
        print(f"  - Symbol effects: {'enabled' if apply_symbol_effects else 'disabled'}")
        print(f"  - Image effects: {'enabled' if apply_image_effects else 'disabled'}")

        for i in range(num_images):
            image_id = i + 1
            
            # Randomly vary grid dimensions and cell size
            current_rows = random.randint(rows[0], rows[1])
            current_cols = random.randint(cols[0], cols[1])
            current_cell_size = random.randint(cell_size[0], cell_size[1])
            
            # Generate building with symbols
            floor_plan_img, annotations, grid, rooms = generate_building_with_symbols(
                rows=current_rows,
                cols=current_cols,
                cell_size=current_cell_size,
                door_size=door_size,
                min_room_width=min_room_width,
                min_building_area_ratio=min_building_area_ratio,
                symbols_dir=self.symbols_dir,
                symbols_per_room=symbols_per_room,
                scale_range=scale_range,
                rotation_range=rotation_range,
                symbol_classes=symbol_classes,
                show_labels=show_labels,
                apply_symbol_effects=apply_symbol_effects,
            )

            # Apply image effects (noise, brightness, contrast, color shifts)
            if apply_image_effects:
                floor_plan_img = self._apply_image_effects(floor_plan_img)

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
                f"[{i+1}/{num_images}] Generated {filename} ({current_rows}x{current_cols}) - "
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
