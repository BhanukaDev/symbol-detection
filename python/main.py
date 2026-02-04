import cv2
import json
from floor_grid import (
    generate_building_grid,
    get_floor_image,
    generate_building_with_symbols,
)


def main():
    # Parameters
    rows = 36
    cols = 30
    door_size = 2
    min_room_width = 8
    min_building_area_ratio = 0.6
    cell_size = 20

    print("Generating building layout with symbols...")

    # Generate building with symbols
    floor_plan_img, annotations, grid, rooms = generate_building_with_symbols(
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        door_size=door_size,
        min_room_width=min_room_width,
        min_building_area_ratio=min_building_area_ratio,
        symbols_dir="data/electrical-symbols",
        symbols_per_room=(1, 4),  # 1 to 4 symbols per room
        scale_range=(0.8, 1.5),  # Scale symbols between 80% and 150%
        rotation_range=(0.0, 360.0),  # Random rotation between 0 and 360 degrees
        show_labels=True,
    )

    print(f"Generated {len(rooms)} rooms with {len(annotations)} symbols placed.")

    # Print symbol placement info
    for ann in annotations:
        print(
            f"  - {ann['class_name']} at ({ann['x']}, {ann['y']}) "
            f"size: {ann['width']}x{ann['height']} (scale: {ann['scale']:.2f}, rot: {ann['rotation']:.1f}°)"
        )

    # Save the result
    output_filename = "final_floor_plan_with_symbols.png"
    cv2.imwrite(output_filename, floor_plan_img)
    print(f"Floor plan with symbols saved to {output_filename}")

    # Save annotations to JSON (useful for training)
    annotations_filename = "symbol_annotations.json"
    with open(annotations_filename, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotations saved to {annotations_filename}")


def main_basic():
    """Original main function without symbols."""
    # Parameters
    rows = 32
    cols = 48
    door_size = 2
    min_room_width = 8
    min_building_area_ratio = 0.6
    cell_size = 20

    print("Generating building layout...")
    grid, rooms, doors = generate_building_grid(
        rows=rows,
        cols=cols,
        door_size=door_size,
        min_room_width=min_room_width,
        min_building_area_ratio=min_building_area_ratio,
    )

    print(f"Generated {len(rooms)} rooms and {len(doors)} doors.")

    print("Generating floor plan image...")
    floor_plan_img = get_floor_image(
        grid, rooms, doors, cell_size=cell_size, show_labels=True
    )

    # Save the result
    output_filename = "final_floor_plan.png"
    cv2.imwrite(output_filename, floor_plan_img)
    print(f"Floor plan saved to {output_filename}")


if __name__ == "__main__":
    main()
