import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from symbol_detection.inference import SymbolDetectionPredictor


def main():
    parser = argparse.ArgumentParser(description="Run inference on floor plan images")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='detections', help='Output directory for results')
    parser.add_argument('--conf-threshold', type=float, default=0.50, help='Confidence threshold')
    parser.add_argument('--categories', type=str, default=None, help='Path to categories JSON')
    parser.add_argument('--num-classes', type=int, default=8, help='Number of classes (including background)')
    parser.add_argument('--save-vis', action='store_true', help='Save visualized results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = SymbolDetectionPredictor(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        categories_file=args.categories,
        confidence_threshold=args.conf_threshold,
    )
    
    # Get input images
    input_path = Path(args.image)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    
    all_results = []
    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ✗ Failed to load image")
            continue
        
        # Run inference
        detections = predictor.predict(image, conf_threshold=args.conf_threshold)
        
        # Save results
        result = {
            'image': str(img_path),
            'width': image.shape[1],
            'height': image.shape[0],
            'num_detections': len(detections),
            'detections': detections,
        }
        all_results.append(result)
        
        print(f"  ✓ Detected {len(detections)} symbols")
        for i, det in enumerate(detections, 1):
            print(f"    {i}. {det['class_name']} (conf: {det['confidence']:.3f})")
        
        # Save visualization if requested
        if args.save_vis:
            vis_image = predictor.visualize(image, detections)
            vis_path = output_dir / f"{img_path.stem}_detections.png"
            cv2.imwrite(str(vis_path), vis_image)
            print(f"  Saved visualization: {vis_path.name}")
        
        # Save detections JSON
        json_path = output_dir / f"{img_path.stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved detections: {json_path.name}")
        print()
    
    # Save summary
    summary = {
        'total_images': len(image_paths),
        'processed': len(all_results),
        'confidence_threshold': args.conf_threshold,
        'total_detections': sum(r['num_detections'] for r in all_results),
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Complete! Results saved to {output_dir}")
    print(f"  - Total detections: {summary['total_detections']}")


if __name__ == '__main__':
    main()
