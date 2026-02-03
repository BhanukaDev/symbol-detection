import cv2
from floor_grid import generate_building_grid, get_floor_image


def main():
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
