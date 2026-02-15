import cv2
from floor_grid import (
    generate_building_grid,
    get_floor_image,
)
from dataset_generator import generate_coco_dataset


def main():
    """Generate dataset with COCO format annotations."""
    # Dataset generation settings
    num_images = 10
    output_dir = "dataset"

    # Building parameters
    rows = 80
    cols = 120
    door_size = 2
    min_room_width = 8
    min_building_area_ratio = 0.6
    cell_size = 20

    # Generate COCO format dataset
    coco_data = generate_coco_dataset(
        output_dir=output_dir,
        symbols_dir="data/electrical-symbols",
        num_images=num_images,
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        door_size=door_size,
        min_room_width=min_room_width,
        min_building_area_ratio=min_building_area_ratio,
        symbols_per_room=(1, 4),
        scale_range=(0.8, 1.5),
        rotation_range=(0.0, 360.0),
        show_labels=False,
    )

    # Print summary
    print(f"\nDataset summary:")
    print(f"  - Total images: {len(coco_data['images'])}")
    print(f"  - Total annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {[cat['name'] for cat in coco_data['categories']]}")


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
