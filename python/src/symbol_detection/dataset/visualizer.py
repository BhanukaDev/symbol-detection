"""COCO annotations visualizer CLI"""
import cv2
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class COCOVisualizer:
    def __init__(self, annotations_path: str, images_dir: str):
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)

        with open(self.annotations_path, "r") as f:
            self.coco_data = json.load(f)

        self.category_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }
        self.category_colors = {
            cat_id: self._generate_color(i)
            for i, cat_id in enumerate(self.category_id_to_name.keys())
        }
        self.annotations_by_image: Dict[int, list] = {}
        for ann in self.coco_data["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        print(f"Loaded {len(self.coco_data['images'])} images, "
              f"{len(self.coco_data['annotations'])} annotations, "
              f"{len(self.coco_data['categories'])} categories")
        for cat_id, name in self.category_id_to_name.items():
            print(f"  - {name} (ID: {cat_id}) color: {self.category_colors[cat_id]}")

    def _generate_color(self, index: int) -> Tuple[int, int, int]:
        random.seed(index * 12345)
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def visualize_image(
        self,
        image_id: int,
        show_labels: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.5,
        save_path: Optional[str] = None,
    ) -> bool:
        image_info = next(
            (img for img in self.coco_data["images"] if img["id"] == image_id), None
        )
        if image_info is None:
            print(f"Error: Image ID {image_id} not found.")
            return True

        image_path = self.images_dir / image_info["file_name"]
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return True

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image: {image_path}")
            return True

        for ann in self.annotations_by_image.get(image_id, []):
            cat_id = ann["category_id"]
            name = self.category_id_to_name[cat_id]
            color = self.category_colors[cat_id]
            x, y, w, h = (int(v) for v in ann["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, bbox_thickness)
            if show_labels:
                (tw, th), bl = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(img, (x, y - th - bl - 4), (x + tw, y), color, -1)
                cv2.putText(img, name, (x, y - bl - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        annotations = self.annotations_by_image.get(image_id, [])
        cv2.putText(img,
                    f"Image: {image_info['file_name']} | Annotations: {len(annotations)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Saved: {save_path}")
        else:
            cv2.imshow(f"COCO Visualization - Image ID {image_id}", img)
            print(f"\nShowing {image_info['file_name']} with {len(annotations)} annotations")
            print("Press any key to continue, 'q' to quit...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key in (ord("q"), ord("Q")):
                return False

        return True

    def visualize_all(
        self,
        max_images: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        images = self.coco_data["images"]
        if max_images:
            images = images[:max_images]

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            print(f"\nSaving visualizations to: {save_dir}/")

        for i, img_info in enumerate(images):
            print(f"\n[{i + 1}/{len(images)}] {img_info['file_name']}")
            save_path = str(Path(save_dir) / f"viz_{img_info['file_name']}") if save_dir else None
            if not self.visualize_image(img_info["id"], save_path=save_path, **kwargs):
                print("\nVisualization stopped by user.")
                break

        if not save_dir:
            cv2.destroyAllWindows()
        print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO format annotations")
    parser.add_argument("--annotations", type=str, default="dataset/annotations.json",
                        help="Path to COCO annotations JSON file")
    parser.add_argument("--images", type=str, default="dataset/images",
                        help="Directory containing images")
    parser.add_argument("--image-id", type=int, default=None,
                        help="Visualize a specific image ID (default: all)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to show")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save visualizations here instead of displaying")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide category labels on bounding boxes")
    args = parser.parse_args()

    viz = COCOVisualizer(annotations_path=args.annotations, images_dir=args.images)

    if args.image_id is not None:
        save_path = (str(Path(args.save_dir) / f"viz_image_{args.image_id}.png")
                     if args.save_dir else None)
        viz.visualize_image(image_id=args.image_id,
                            show_labels=not args.no_labels,
                            save_path=save_path)
    else:
        viz.visualize_all(max_images=args.max_images,
                          save_dir=args.save_dir,
                          show_labels=not args.no_labels)


if __name__ == "__main__":
    main()
