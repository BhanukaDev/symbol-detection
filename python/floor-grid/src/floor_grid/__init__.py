from typing import Tuple, List, Any, Optional, Dict
from .models import Grid
from .building_generator import BuildingShapeGenerator
from .room_generator import RoomDivider
from .door_generator import DoorWindowGenerator
from .visualizer import FloorPlanVisualizer
from .place_non_overlapping_rect_rooms import (
    place_non_overlapping_rect_rooms,
)  # Keep existing
from .symbol_placer import (
    Symbol,
    PlacedSymbol,
    SymbolLoader,
    SymbolPlacer,
    generate_building_with_symbols,
)


def generate_building_grid(
    rows: int = 32,
    cols: int = 48,
    door_size: int = 2,
    min_room_width: int = 5,
    min_building_area_ratio: float = 0.6,
) -> Tuple[Grid, List[Any], List[Any]]:
    """
    Generates a complete building grid with rooms and doors.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        door_size: Size of doors in cells.
        min_room_width: Minimum width of a room in cells.
        min_building_area_ratio: Target ratio of building area to total grid area.

    Returns:
        tuple: (grid, rooms, doors)
    """
    base_grid = Grid(rows, cols)

    # Create the building shape generator
    shape_gen = BuildingShapeGenerator(
        base_grid,
        door_size=door_size,
        min_room_width=min_room_width,
        min_building_area_ratio=min_building_area_ratio,
    )

    # Generate the building shape
    grid = shape_gen.generate_building_shape()

    # Then divide into rooms
    room_divider = RoomDivider(grid, min_room_width=min_room_width)
    rooms = room_divider.divide_into_rooms()

    # Generate doors
    door_window_gen = DoorWindowGenerator(
        grid, rooms, door_size=door_size, min_room_width=min_room_width
    )
    doors = door_window_gen.generate_doors()

    return grid, rooms, doors


def get_floor_image(
    grid: Grid,
    rooms: List[Any],
    doors: List[Any],
    cell_size: int = 20,
    show_labels: bool = True,
):
    """
    Generates a visual representation of the floor plan.

    Args:
        grid: The grid object.
        rooms: List of Room objects.
        doors: List of Door objects.
        cell_size: Size of each cell in pixels.
        show_labels: Whether to overlay room labels.

    Returns:
        numpy.ndarray: The generated image.
    """
    visualizer = FloorPlanVisualizer(grid, rooms, doors, cell_size=cell_size)

    # Create the floor plan
    floor_plan_img = visualizer.create_floor_plan()

    # Optionally add room labels
    if show_labels:
        floor_plan_img = visualizer.add_room_labels(floor_plan_img)

    return floor_plan_img


__all__ = [
    "place_non_overlapping_rect_rooms",
    "generate_building_grid",
    "get_floor_image",
    "generate_building_with_symbols",
    "Symbol",
    "PlacedSymbol",
    "SymbolLoader",
    "SymbolPlacer",
]
