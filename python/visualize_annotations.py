"""
COCO Annotations Visualizer

Visualizes bounding boxes from COCO format annotations on images
to verify that annotations are correct.
"""

import cv2
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class COCOVisualizer:
    """Visualizes COCO format annotations on images."""

    def __init__(self, annotations_path: str, images_dir: str):
        """
        Initialize the visualizer.

        Args:
            annotations_path: Path to COCO format annotations.json file.
            images_dir: Directory containing the images.
        """
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        
        # Load COCO data
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup dictionaries
        self.category_id_to_name = {
            cat['id']: cat['name'] 
            for cat in self.coco_data['categories']
        }
        
        # Generate random colors for each category
        self.category_colors = {
            cat_id: self._generate_color(i)
            for i, cat_id in enumerate(self.category_id_to_name.keys())
        }
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        print(f"Loaded COCO data:")
        print(f"  - Images: {len(self.coco_data['images'])}")
        print(f"  - Annotations: {len(self.coco_data['annotations'])}")
        print(f"  - Categories: {len(self.coco_data['categories'])}")
        print(f"\nCategories:")
        for cat_id, cat_name in self.category_id_to_name.items():
            color = self.category_colors[cat_id]
            print(f"  - {cat_name} (ID: {cat_id}) - Color: RGB{color}")

    def _generate_color(self, index: int) -> Tuple[int, int, int]:
        """Generate a distinct color for a category."""
        random.seed(index * 12345)  # Deterministic colors
        return (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )

    def visualize_image(
        self,
        image_id: int,
        show_labels: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.5,
        save_path: Optional[str] = None
    ) -> bool:
        """
        Visualize annotations for a specific image.

        Args:
            image_id: ID of the image to visualize.
            show_labels: Whether to show category labels on bounding boxes.
            bbox_thickness: Thickness of bounding box lines.
            font_scale: Scale of the text font.
            save_path: If provided, save the visualized image to this path.
        """
        # Find image info
        image_info = None
        for img in self.coco_data['images']:
            if img['id'] == image_id:
                image_info = img
                break
        
        if image_info is None:
            print(f"Error: Image ID {image_id} not found.")
            return True
        
        # Load image
        image_path = self.images_dir / image_info['file_name']
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return True
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image: {image_path}")
            return True
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(image_id, [])
        
        # Draw bounding boxes
        for ann in annotations:
            category_id = ann['category_id']
            category_name = self.category_id_to_name[category_id]
            color = self.category_colors[category_id]
            
            # Get bbox (COCO format: [x, y, width, height])
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, bbox_thickness)
            
            # Draw label
            if show_labels:
                label = f"{category_name}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    img,
                    (x, y - text_height - baseline - 4),
                    (x + text_width, y),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    img,
                    label,
                    (x, y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        # Add info text
        info_text = f"Image: {image_info['file_name']} | Annotations: {len(annotations)}"
        cv2.putText(
            img,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Save or display
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Saved visualization to: {save_path}")
        else:
            # Display image
            window_name = f"COCO Visualization - Image ID {image_id}"
            cv2.imshow(window_name, img)
            print(f"\nShowing {image_info['file_name']} with {len(annotations)} annotations")
            print("Press any key to continue, 'q' to quit...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == ord('q') or key == ord('Q'):
                return False
        
        return True

    def visualize_all(
        self,
        max_images: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Visualize all images in the dataset.

        Args:
            max_images: Maximum number of images to visualize (None for all).
            save_dir: If provided, save visualizations to this directory.
            **kwargs: Additional arguments passed to visualize_image().
        """
        images_to_visualize = self.coco_data['images']
        if max_images:
            images_to_visualize = images_to_visualize[:max_images]
        
        # Create save directory if needed
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving visualizations to: {save_dir_path}/")
        
        for i, img_info in enumerate(images_to_visualize):
            print(f"\n[{i+1}/{len(images_to_visualize)}] Processing {img_info['file_name']}...")
            
            save_path = None
            if save_dir:
                save_path = str(Path(save_dir) / f"viz_{img_info['file_name']}")
            
            should_continue = self.visualize_image(
                img_info['id'],
                save_path=save_path,
                **kwargs
            )
            
            if should_continue is False:
                print("\nVisualization stopped by user.")
                break
        
        if not save_dir:
            cv2.destroyAllWindows()
        
        print("\nVisualization complete!")


def main():
    """Main function to run the visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize COCO format annotations')
    parser.add_argument(
        '--annotations',
        type=str,
        default='dataset/annotations.json',
        help='Path to COCO annotations JSON file'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='dataset/images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--image-id',
        type=int,
        default=None,
        help='Visualize specific image ID (default: show all)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to visualize'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Save visualizations to this directory instead of displaying'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide category labels on bounding boxes'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = COCOVisualizer(
        annotations_path=args.annotations,
        images_dir=args.images
    )
    
    # Visualize
    if args.image_id is not None:
        # Visualize specific image
        save_path = None
        if args.save_dir:
            save_path = str(Path(args.save_dir) / f"viz_image_{args.image_id}.png")
        
        visualizer.visualize_image(
            image_id=args.image_id,
            show_labels=not args.no_labels,
            save_path=save_path
        )
    else:
        # Visualize all images
        visualizer.visualize_all(
            max_images=args.max_images,
            save_dir=args.save_dir,
            show_labels=not args.no_labels
        )


if __name__ == "__main__":
    main()
