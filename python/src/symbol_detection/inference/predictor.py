import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class SymbolDetectionPredictor:
    """Production inference pipeline for electrical symbol detection."""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        num_classes: int = 7,
        categories_file: Optional[str | Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.50,
        nms_threshold: float = 0.5,
    ):
        """
        Initialize the predictor with a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            num_classes: Number of object classes
            categories_file: JSON file with category names (from dataset)
            device: Device to use ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence to keep detections
            nms_threshold: NMS threshold for duplicate suppression
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model from {self.checkpoint_path}...")
        self.model = self._load_model()
        
        # Load category names if available
        self.categories = self._load_categories(categories_file)
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Confidence threshold: {self.confidence_threshold}")

    def _load_model(self):
        """Load FasterRCNN model from checkpoint."""
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=self.num_classes,
            trainable_backbone_layers=4,
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle checkpoint format: extract model_state_dict if it's wrapped
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        return model

    def _load_categories(self, categories_file: Optional[str | Path]) -> Dict[int, str]:
        """Load category names from COCO format JSON.
        
        Maps 0-based indices (used by model) to category names.
        Index 0 is always background, indices 1+ map to COCO categories in order.
        """
        if categories_file is None:
            return {0: "Background", **{i: f"Symbol_{i}" for i in range(1, self.num_classes)}}
        
        try:
            with open(categories_file, 'r') as f:
                data = json.load(f)
            
            # Create 0-based index mapping (same as training: enumerate categories)
            categories = {0: "Background"}
            for idx, cat in enumerate(data['categories'], start=1):
                categories[idx] = cat['name']
            
            return categories
        except Exception as e:
            print(f"⚠ Could not load categories: {e}")
            return {0: "Background", **{i: f"Symbol_{i}" for i in range(1, self.num_classes)}}

    def preprocess(
        self,
        image: np.ndarray,
        target_size: int = 512,
        letterbox: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR or grayscale)
            target_size: Target image size
            letterbox: If True, preserve aspect ratio with padding
        
        Returns:
            Preprocessed tensor and metadata dict
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        metadata = {'original_size': (h, w)}
        
        if letterbox:
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            y_off = (target_size - new_h) // 2
            x_off = (target_size - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            metadata['scale'] = scale
            metadata['offset'] = (x_off, y_off)
            image = canvas
        else:
            image = cv2.resize(image, (target_size, target_size))
            metadata['scale'] = target_size / max(h, w)
            metadata['offset'] = (0, 0)
        
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), metadata

    def postprocess(
        self,
        predictions: Dict,
        metadata: Dict,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Model output dict
            metadata: Preprocessing metadata
            conf_threshold: Override confidence threshold
        
        Returns:
            List of detections with boxes, classes, and scores
        """
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        
        boxes = predictions['boxes'].cpu().detach().numpy()
        scores = predictions['scores'].cpu().detach().numpy()
        labels = predictions['labels'].cpu().detach().numpy()
        
        scale = metadata.get('scale', 1.0)
        x_off, y_off = metadata.get('offset', (0, 0))
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            
            x1, y1, x2, y2 = box
            
            # Undo letterbox transform
            x1 = (x1 - x_off) / scale
            y1 = (y1 - y_off) / scale
            x2 = (x2 - x_off) / scale
            y2 = (y2 - y_off) / scale
            
            # Clip to original image bounds
            orig_h, orig_w = metadata['original_size']
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': int(label),
                'class_name': self.categories.get(int(label), f"Unknown"),
                'confidence': float(score),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
            })
        
        return detections

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (BGR or grayscale)
            conf_threshold: Override confidence threshold
        
        Returns:
            List of detected symbols
        """
        with torch.no_grad():
            tensor, metadata = self.preprocess(image)
            # tensor has shape [1, C, H, W], squeeze to [C, H, W] for model input
            tensor_3d = tensor.squeeze(0)
            predictions = self.model([tensor_3d])[0]
            detections = self.postprocess(predictions, metadata, conf_threshold)
        
        return detections

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> List[List[Dict]]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of input images
            conf_threshold: Override confidence threshold
        
        Returns:
            List of detection lists
        """
        all_detections = []
        
        with torch.no_grad():
            tensors = []
            metadata_list = []
            
            for image in images:
                tensor, metadata = self.preprocess(image)
                tensors.append(tensor.squeeze(0))
                metadata_list.append(metadata)
            
            tensor_batch = torch.stack(tensors).to(self.device)
            predictions = self.model(tensor_batch)
            
            for pred, metadata in zip(predictions, metadata_list):
                detections = self.postprocess(pred, metadata, conf_threshold)
                all_detections.append(detections)
        
        return all_detections

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image (BGR)
            detections: List of detections
            thickness: Box thickness
            font_scale: Font scale for labels
        
        Returns:
            Annotated image
        """
        # Create copy to avoid modifying original
        vis_image = image.copy()
        
        # Color palette for different classes
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
        ]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            class_id = det['class_id']
            conf = det['confidence']
            class_name = det['class_name']
            
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y1 - 5, label_size[1] + 5)
            
            cv2.rectangle(
                vis_image,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1,
            )
            cv2.putText(
                vis_image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )
        
        return vis_image
