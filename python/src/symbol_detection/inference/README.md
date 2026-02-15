# Inference Module

Production-ready inference pipeline for electrical symbol detection.

## Quick Start

### Python API

```python
from symbol_detection.inference import SymbolDetectionPredictor
import cv2

# Initialize predictor
predictor = SymbolDetectionPredictor(
    checkpoint_path='model_epoch_final.pth',
    num_classes=7,
    confidence_threshold=0.50
)

# Load image
image = cv2.imread('floor_plan.png')

# Run inference
detections = predictor.predict(image)

# Visualize
annotated = predictor.visualize(image, detections)
cv2.imshow('Detections', annotated)
cv2.waitKey(0)
```

### Command Line

```bash
run-inference \
    --checkpoint model_epoch_final.pth \
    --image /path/to/image.png \
    --output ./detections \
    --conf-threshold 0.50 \
    --save-vis
```

## Features

- **Fast Inference**: ~50-100ms per image on A100 GPU
- **Batch Processing**: Process multiple images efficiently
- **Flexible Thresholds**: Adjust confidence thresholds per inference
- **Visualization**: Built-in visualization with bounding boxes
- **Production Ready**: Error handling, logging, JSON output

## Configuration

### Confidence Threshold

- **0.30**: High recall, catch all symbols (~95%), accept false positives
- **0.50**: Balanced (recommended) (~90% precision)
- **0.70**: High precision, only very confident detections (~95%)

### Batch Size

- Single image: `batch_size=1`
- Multiple images: `batch_size=4-8` (adjust based on GPU memory)
- A100 40GB: Can safely handle `batch_size=16` for inference

## Output Format

Detections are returned as a list of dictionaries:

```python
[
    {
        'bbox': [x1, y1, x2, y2],        # Bounding box in original image coords
        'class_id': 0,                    # Category ID
        'class_name': 'Light',            # Category name
        'confidence': 0.95,               # Detection confidence [0-1]
        'width': 50.0,                    # Box width
        'height': 45.0,                   # Box height
    },
    ...
]
```

## Performance

On NVIDIA A100 (40GB):
- Single image inference: ~50-100ms
- Throughput: ~10-20 images/second
- GPU memory usage: ~4GB

On CPU:
- Single image inference: ~500-1000ms
- Not recommended for real-time applications

## Integration Examples

### Web API (Flask)

```python
from flask import Flask, request, jsonify
from symbol_detection.inference import SymbolDetectionPredictor
import cv2
import numpy as np

app = Flask(__name__)
predictor = SymbolDetectionPredictor('model_epoch_final.pth')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    detections = predictor.predict(image)
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Batch Processing

```python
from pathlib import Path
import json

image_dir = Path('floor_plans/')
predictor = SymbolDetectionPredictor('model_epoch_final.pth')

for img_path in image_dir.glob('*.png'):
    image = cv2.imread(str(img_path))
    detections = predictor.predict(image)
    
    # Save results
    with open(img_path.with_suffix('.json'), 'w') as f:
        json.dump({'detections': detections}, f)
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use CPU instead: `device='cpu'`
- Resize images before inference

### Low Detection Quality
- Lower confidence threshold
- Check model checkpoint path
- Ensure input image format is correct (BGR for OpenCV)

### Slow Inference
- Use GPU: `device='cuda'`
- Batch process images instead of single inference
- Reduce image resolution

## See Also

- [Training](../training/README.md) - Model training guide
- [Dataset](../dataset/README.md) - Dataset generation
