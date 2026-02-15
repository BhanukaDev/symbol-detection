"""
Symbol Placer Module

This module handles loading electrical symbols and placing them
inside building rooms without overlapping.
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from .models import Grid, Room

try:
    from effects import water_wave_distortion, twirl_distortion
except ImportError:
    water_wave_distortion = None
    twirl_distortion = None


@dataclass
class Symbol:
    """Represents an electrical symbol."""

    name: str
    image: np.ndarray
    width: int
    height: int
    variant_path: str  # path to source image file


@dataclass
class PlacedSymbol:
    """Represents a symbol placed in the building."""

    symbol: Symbol
    x: int
    y: int
    room_index: int
    scale: float
    rotation: float = 0.0


class SymbolLoader:
    """Loads electrical symbols from a directory structure."""

    def __init__(self, symbols_dir: str):
        """
        Initialize the symbol loader.

        Args:
            symbols_dir: Path to the directory containing symbol folders.
                         Each subfolder represents a symbol class and contains
                         variant images (PNG files).
        """
        self.symbols_dir = Path(symbols_dir)
        self.symbol_classes: Dict[str, List[Symbol]] = {}
        self._load_symbols()

    def _load_symbols(self):
        """Load all symbols from the directory structure."""
        if not self.symbols_dir.exists():
            raise ValueError(f"Symbols directory does not exist: {self.symbols_dir}")

        for symbol_folder in self.symbols_dir.iterdir():
            if symbol_folder.is_dir():
                symbol_name = symbol_folder.name
                self.symbol_classes[symbol_name] = []

                # Load all PNG files in this folder
                for img_file in symbol_folder.glob("*.png"):
                    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        symbol = Symbol(
                            name=symbol_name,
                            image=img,
                            width=img.shape[1],
                            height=img.shape[0],
                            variant_path=str(img_file),
                        )
                        self.symbol_classes[symbol_name].append(symbol)

        print(f"Loaded {len(self.symbol_classes)} symbol classes:")
        for name, variants in self.symbol_classes.items():
            print(f"  - {name}: {len(variants)} variant(s)")

    def get_random_symbol(self, symbol_class: Optional[str] = None) -> Optional[Symbol]:
        """
        Get a random symbol, optionally from a specific class.

        Args:
            symbol_class: If provided, get a random variant from this class.
                         If None, get a random symbol from any class.

        Returns:
            A random Symbol, or None if no symbols are available.
        """
        if symbol_class:
            if (
                symbol_class in self.symbol_classes
                and self.symbol_classes[symbol_class]
            ):
                return random.choice(self.symbol_classes[symbol_class])
            return None

        # Get random from any class
        all_symbols = []
        for variants in self.symbol_classes.values():
            all_symbols.extend(variants)

        if all_symbols:
            return random.choice(all_symbols)
        return None

    def get_all_symbol_classes(self) -> List[str]:
        """Get list of all available symbol class names."""
        return list(self.symbol_classes.keys())


class SymbolPlacer:
    """Places symbols inside building rooms without overlapping."""

    def __init__(
        self,
        grid: Grid,
        rooms: List[Room],
        cell_size: int = 20,
        wall_thickness: int = 8,
        symbol_padding: int = 5,
        apply_symbol_effects: bool = True,
    ):
        """
        Initialize the symbol placer.

        Args:
            grid: The building grid.
            rooms: List of Room objects.
            cell_size: Size of each cell in pixels.
            wall_thickness: Thickness of walls in pixels.
            symbol_padding: Minimum padding between symbols and walls/other symbols.
            apply_symbol_effects: Whether to apply distortion effects to symbols.
        """
        self.grid = grid
        self.rooms = rooms
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.symbol_padding = symbol_padding
        self.apply_symbol_effects = apply_symbol_effects
        self.placed_symbols: List[PlacedSymbol] = []

        # Calculate image dimensions
        self.img_width = grid.cols * cell_size
        self.img_height = grid.rows * cell_size

    def _get_room_bounds_pixels(self, room: Room) -> Tuple[int, int, int, int]:
        """
        Get the pixel bounds of a room, accounting for wall thickness.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixels.
        """
        bounds = room.get_bounds()
        if not bounds:
            return (0, 0, 0, 0)

        min_r, max_r, min_c, max_c = bounds

        # Convert to pixels and account for wall thickness
        half_wall = self.wall_thickness // 2
        padding = half_wall + self.symbol_padding

        x_min = min_c * self.cell_size + padding
        y_min = min_r * self.cell_size + padding
        x_max = (max_c + 1) * self.cell_size - padding
        y_max = (max_r + 1) * self.cell_size - padding

        return (x_min, y_min, x_max, y_max)

    def _check_overlap(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        existing_symbols: List[PlacedSymbol],
    ) -> bool:
        """
        Check if a rectangle overlaps with any existing placed symbols.

        Args:
            x, y: Top-left corner of the rectangle.
            width, height: Dimensions of the rectangle.
            existing_symbols: List of already placed symbols.

        Returns:
            True if there is an overlap, False otherwise.
        """
        for placed in existing_symbols:
            # Add padding to both symbols when checking overlap
            p = self.symbol_padding

            # Get the bounding box dimensions of the placed symbol (accounting for rotation)
            placed_width, placed_height = self._get_rotated_bbox_size(
                placed.symbol.width * placed.scale,
                placed.symbol.height * placed.scale,
                placed.rotation,
            )

            # Check for overlap with padding
            if not (
                x + width + p <= placed.x
                or placed.x + placed_width + p <= x
                or y + height + p <= placed.y
                or placed.y + placed_height + p <= y
            ):
                return True

        return False

    def _get_rotated_bbox_size(
        self, width: float, height: float, angle_degrees: float
    ) -> Tuple[int, int]:
        """
        Calculate the bounding box size of a rotated rectangle.

        Args:
            width: Original width.
            height: Original height.
            angle_degrees: Rotation angle in degrees.

        Returns:
            Tuple of (new_width, new_height) for the bounding box.
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))

        new_width = int(np.ceil(width * cos_a + height * sin_a))
        new_height = int(np.ceil(width * sin_a + height * cos_a))

        return new_width, new_height

    def _rotate_image(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Rotate an image by a given angle, expanding the canvas to fit.

        Args:
            image: Input image (can have alpha channel).
            angle_degrees: Rotation angle in degrees (counter-clockwise).

        Returns:
            Rotated image with expanded canvas.
        """
        if angle_degrees == 0:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        # Calculate new bounding box size
        new_w, new_h = self._get_rotated_bbox_size(w, h, angle_degrees)

        # Adjust the rotation matrix to account for the new canvas size
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Determine border value based on whether image has alpha channel
        if image.shape[2] == 4:
            border_value = (0, 0, 0, 0)  # Transparent
        else:
            border_value = (255, 255, 255)  # White background

        # Apply the rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        return rotated

    def _is_point_in_room(self, x: int, y: int, room: Room) -> bool:
        """
        Check if a pixel point is inside a room.

        Args:
            x, y: Pixel coordinates.
            room: Room to check against.

        Returns:
            True if the point is in the room.
        """
        cell_c = x // self.cell_size
        cell_r = y // self.cell_size
        return (cell_r, cell_c) in room.cells

    def _is_rect_in_room(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        room: Room,
    ) -> bool:
        """
        Check if a rectangle is fully inside a room.

        Args:
            x, y: Top-left corner in pixels.
            width, height: Dimensions in pixels.
            room: Room to check against.

        Returns:
            True if the entire rectangle is inside the room.
        """
        # Check all four corners
        corners = [
            (x, y),
            (x + width - 1, y),
            (x, y + height - 1),
            (x + width - 1, y + height - 1),
        ]

        for cx, cy in corners:
            if not self._is_point_in_room(cx, cy, room):
                return False

        return True

    def place_symbol_in_room(
        self,
        symbol: Symbol,
        room_index: int,
        scale: float = 1.0,
        rotation: float = 0.0,
        max_attempts: int = 100,
    ) -> Optional[PlacedSymbol]:
        """
        Try to place a symbol in a specific room.

        Args:
            symbol: The symbol to place.
            room_index: Index of the room to place the symbol in.
            scale: Scale factor to apply to the symbol.
            rotation: Rotation angle in degrees (counter-clockwise).
            max_attempts: Maximum number of random placement attempts.

        Returns:
            PlacedSymbol if successful, None if placement failed.
        """
        if room_index < 0 or room_index >= len(self.rooms):
            return None

        room = self.rooms[room_index]
        bounds = self._get_room_bounds_pixels(room)
        x_min, y_min, x_max, y_max = bounds

        # Calculate scaled dimensions
        scaled_width = int(symbol.width * scale)
        scaled_height = int(symbol.height * scale)

        # Calculate bounding box size after rotation
        bbox_width, bbox_height = self._get_rotated_bbox_size(
            scaled_width, scaled_height, rotation
        )

        # Check if symbol fits in room bounds at all
        available_width = x_max - x_min - bbox_width
        available_height = y_max - y_min - bbox_height

        if available_width < 0 or available_height < 0:
            return None  # Symbol too big for room

        # Try random positions
        for _ in range(max_attempts):
            x = random.randint(x_min, x_min + available_width)
            y = random.randint(y_min, y_min + available_height)

            # Check if the rectangle is fully inside the room
            if not self._is_rect_in_room(x, y, bbox_width, bbox_height, room):
                continue

            # Check for overlap with existing symbols
            if not self._check_overlap(
                x, y, bbox_width, bbox_height, self.placed_symbols
            ):
                placed = PlacedSymbol(
                    symbol=symbol,
                    x=x,
                    y=y,
                    room_index=room_index,
                    scale=scale,
                    rotation=rotation,
                )
                self.placed_symbols.append(placed)
                return placed

        return None

    def place_symbols_randomly(
        self,
        symbol_loader: SymbolLoader,
        symbols_per_room: Tuple[int, int] = (1, 5),
        scale_range: Tuple[float, float] = (0.8, 1.5),
        rotation_range: Tuple[float, float] = (0.0, 360.0),
        symbol_classes: Optional[List[str]] = None,
    ) -> List[PlacedSymbol]:
        """
        Place random symbols in all rooms.

        Args:
            symbol_loader: SymbolLoader instance with loaded symbols.
            symbols_per_room: (min, max) number of symbols to place per room.
            scale_range: (min, max) scale factor range for symbols.
            rotation_range: (min, max) rotation angle range in degrees.
            symbol_classes: List of symbol classes to use, or None for all.

        Returns:
            List of all successfully placed symbols.
        """
        available_classes = symbol_classes or symbol_loader.get_all_symbol_classes()

        for room_idx, room in enumerate(self.rooms):
            # Skip very small rooms
            if room.get_area() < 4:
                continue

            num_symbols = random.randint(symbols_per_room[0], symbols_per_room[1])

            for _ in range(num_symbols):
                # Get a random symbol
                symbol_class = random.choice(available_classes)
                symbol = symbol_loader.get_random_symbol(symbol_class)

                if symbol is None:
                    continue

                # Random scale
                scale = random.uniform(scale_range[0], scale_range[1])

                # Random rotation
                rotation = random.uniform(rotation_range[0], rotation_range[1])

                # Try to place it
                self.place_symbol_in_room(symbol, room_idx, scale, rotation)

        return self.placed_symbols

    def _apply_symbol_effects(self, img: np.ndarray) -> np.ndarray:
        """
        Apply distortion effects to a symbol image.

        Args:
            img: Input image (symbol).

        Returns:
            Processed image with effects applied.
        """
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
            print(f"Warning: Symbol effect application failed: {e}")
        
        return img

    def render_symbols_on_image(
        self,
        floor_plan_img: np.ndarray,
        use_alpha: bool = True,
    ) -> np.ndarray:
        """
        Render all placed symbols onto the floor plan image.

        Args:
            floor_plan_img: The floor plan image to draw on.
            use_alpha: Whether to use alpha channel for transparency.

        Returns:
            The image with symbols rendered on it.
        """
        result = floor_plan_img.copy()

        for placed in self.placed_symbols:
            symbol = placed.symbol
            scale = placed.scale
            rotation = placed.rotation

            # Scale the symbol image
            scaled_width = int(symbol.width * scale)
            scaled_height = int(symbol.height * scale)

            if scaled_width <= 0 or scaled_height <= 0:
                continue

            scaled_img = cv2.resize(
                symbol.image,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )

            # Apply symbol effects if enabled
            if self.apply_symbol_effects:
                scaled_img = self._apply_symbol_effects(scaled_img)

            # Apply rotation
            rotated_img = self._rotate_image(scaled_img, rotation)
            rot_height, rot_width = rotated_img.shape[:2]

            x, y = placed.x, placed.y

            # Ensure we don't go out of bounds
            x_end = min(x + rot_width, result.shape[1])
            y_end = min(y + rot_height, result.shape[0])

            if x >= result.shape[1] or y >= result.shape[0]:
                continue

            # Crop symbol if it would go out of bounds
            symbol_x_end = x_end - x
            symbol_y_end = y_end - y

            if symbol_x_end <= 0 or symbol_y_end <= 0:
                continue

            # Handle alpha channel if present
            if use_alpha and rotated_img.shape[2] == 4:
                # Extract alpha channel
                alpha = rotated_img[:symbol_y_end, :symbol_x_end, 3] / 255.0
                alpha = alpha[:, :, np.newaxis]

                # Extract RGB channels
                symbol_rgb = rotated_img[:symbol_y_end, :symbol_x_end, :3]

                # Get the region of interest from the result
                roi = result[y:y_end, x:x_end]

                # Blend using alpha
                blended = (symbol_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
                result[y:y_end, x:x_end] = blended
            else:
                # No alpha, just copy (handle both 3 and 4 channel images)
                if rotated_img.shape[2] == 4:
                    symbol_rgb = rotated_img[:symbol_y_end, :symbol_x_end, :3]
                else:
                    symbol_rgb = rotated_img[:symbol_y_end, :symbol_x_end]

                result[y:y_end, x:x_end] = symbol_rgb

        return result

    def get_symbol_annotations(self) -> List[Dict]:
        """
        Get annotations for all placed symbols (useful for training data).

        Returns:
            List of annotation dictionaries with bounding boxes and class names.
        """
        annotations = []

        for placed in self.placed_symbols:
            scaled_width = int(placed.symbol.width * placed.scale)
            scaled_height = int(placed.symbol.height * placed.scale)

            # Get bounding box size after rotation
            bbox_width, bbox_height = self._get_rotated_bbox_size(
                scaled_width, scaled_height, placed.rotation
            )

            annotation = {
                "class_name": placed.symbol.name,
                "x": placed.x,
                "y": placed.y,
                "width": bbox_width,
                "height": bbox_height,
                "x_center": placed.x + bbox_width // 2,
                "y_center": placed.y + bbox_height // 2,
                "room_index": placed.room_index,
                "scale": placed.scale,
                "rotation": placed.rotation,
                "original_width": scaled_width,
                "original_height": scaled_height,
            }
            annotations.append(annotation)

        return annotations

    def clear_placed_symbols(self):
        """Clear all placed symbols."""
        self.placed_symbols = []


def generate_building_with_symbols(
    rows: int = 32,
    cols: int = 48,
    cell_size: int = 20,
    door_size: int = 2,
    min_room_width: int = 8,
    min_building_area_ratio: float = 0.6,
    symbols_dir: str = "data/electrical-symbols",
    symbols_per_room: Tuple[int, int] = (1, 5),
    scale_range: Tuple[float, float] = (0.8, 1.5),
    rotation_range: Tuple[float, float] = (0.0, 360.0),
    symbol_classes: Optional[List[str]] = None,
    show_labels: bool = False,
    apply_symbol_effects: bool = True,
) -> Tuple[np.ndarray, List[Dict], Grid, List[Room]]:
    """
    Generate a complete building with symbols placed inside.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        cell_size: Size of each cell in pixels.
        door_size: Size of doors in cells.
        min_room_width: Minimum width of a room in cells.
        min_building_area_ratio: Target ratio of building area to total grid area.
        symbols_dir: Path to the directory containing symbol images.
        symbols_per_room: (min, max) number of symbols per room.
        scale_range: (min, max) scale factor range for symbols.
        rotation_range: (min, max) rotation angle range in degrees.
        symbol_classes: List of symbol classes to use, or None for all.
        show_labels: Whether to show room labels on the image.
        apply_symbol_effects: Whether to apply distortion effects to symbols.

    Returns:
        Tuple of (image, annotations, grid, rooms)
        - image: The floor plan with symbols rendered
        - annotations: List of symbol annotation dictionaries
        - grid: The Grid object
        - rooms: List of Room objects
    """
    # Import here to avoid circular imports
    from . import generate_building_grid, get_floor_image

    # Generate the building
    grid, rooms, doors = generate_building_grid(
        rows=rows,
        cols=cols,
        door_size=door_size,
        min_room_width=min_room_width,
        min_building_area_ratio=min_building_area_ratio,
    )

    # Get the base floor plan image
    floor_plan_img = get_floor_image(
        grid, rooms, doors, cell_size=cell_size, show_labels=show_labels
    )

    # Load symbols
    symbol_loader = SymbolLoader(symbols_dir)

    # Create symbol placer and place symbols
    placer = SymbolPlacer(
        grid=grid,
        rooms=rooms,
        cell_size=cell_size,
        apply_symbol_effects=apply_symbol_effects,
    )

    placer.place_symbols_randomly(
        symbol_loader=symbol_loader,
        symbols_per_room=symbols_per_room,
        scale_range=scale_range,
        rotation_range=rotation_range,
        symbol_classes=symbol_classes,
    )

    # Render symbols on the image
    result_img = placer.render_symbols_on_image(floor_plan_img)

    # Get annotations
    annotations = placer.get_symbol_annotations()

    return result_img, annotations, grid, rooms
