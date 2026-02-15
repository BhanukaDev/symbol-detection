"""Dataset generation CLI"""
import argparse
from symbol_detection.dataset.generator import generate_coco_dataset
from symbol_detection.config import DATASET_SETTINGS, SYMBOLS_DIR


def main():
    """Generate dataset with COCO format annotations."""
    parser = argparse.ArgumentParser(description="Generate floor plan dataset")
    parser.add_argument(
        "--num-images",
        type=int,
        default=DATASET_SETTINGS["num_images"],
        help="Number of images to generate",
    )
    parser.add_argument(
        "--rows-min",
        type=int,
        default=DATASET_SETTINGS["rows"][0],
        help="Minimum rows in grid",
    )
    parser.add_argument(
        "--rows-max",
        type=int,
        default=DATASET_SETTINGS["rows"][1],
        help="Maximum rows in grid",
    )
    parser.add_argument(
        "--cols-min",
        type=int,
        default=DATASET_SETTINGS["cols"][0],
        help="Minimum columns in grid",
    )
    parser.add_argument(
        "--cols-max",
        type=int,
        default=DATASET_SETTINGS["cols"][1],
        help="Maximum columns in grid",
    )
    parser.add_argument(
        "--cell-size-min",
        type=int,
        default=DATASET_SETTINGS["cell_size"][0],
        help="Minimum cell size in pixels",
    )
    parser.add_argument(
        "--cell-size-max",
        type=int,
        default=DATASET_SETTINGS["cell_size"][1],
        help="Maximum cell size in pixels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--no-symbol-effects",
        action="store_true",
        help="Disable symbol effects (water waves, twirl)",
    )
    parser.add_argument(
        "--no-image-effects",
        action="store_true",
        help="Disable image effects (noise, brightness, color)",
    )

    args = parser.parse_args()

    # Generate COCO format dataset
    coco_data = generate_coco_dataset(
        output_dir=args.output_dir,
        symbols_dir=str(SYMBOLS_DIR),
        num_images=args.num_images,
        rows=(args.rows_min, args.rows_max),
        cols=(args.cols_min, args.cols_max),
        cell_size=(args.cell_size_min, args.cell_size_max),
        door_size=2,
        min_room_width=8,
        min_building_area_ratio=0.6,
        symbols_per_room=(1, 4),
        scale_range=(0.8, 1.5),
        rotation_range=(0.0, 360.0),
        show_labels=False,
        apply_symbol_effects=not args.no_symbol_effects,
        apply_image_effects=not args.no_image_effects,
    )

    # Print summary
    print(f"\nDataset summary:")
    print(f"  - Total images: {len(coco_data['images'])}")
    print(f"  - Total annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {[cat['name'] for cat in coco_data['categories']]}")


if __name__ == "__main__":
    main()
