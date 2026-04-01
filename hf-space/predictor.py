"""Production inference pipeline for electrical symbol detection."""

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
        num_classes: int = 8,
        categories_file: Optional[str | Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.50,
        nms_threshold: float = 0.5,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {self.checkpoint_path}...")
        self.model = self._load_model()
        self.categories = self._load_categories(categories_file)
        print(f"Model loaded on {self.device}")

    def _load_model(self):
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=self.num_classes,
            trainable_backbone_layers=4,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_categories(self, categories_file: Optional[str | Path]) -> Dict[int, str]:
        if categories_file is None:
            return {i: f"Symbol_{i}" for i in range(self.num_classes)}
        try:
            with open(categories_file, "r") as f:
                data = json.load(f)
            return {cat["id"]: cat["name"] for cat in data["categories"]}
        except Exception:
            return {i: f"Symbol_{i}" for i in range(self.num_classes)}

    def preprocess(self, image: np.ndarray, target_size: int = 512) -> Tuple[torch.Tensor, Dict]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_off = (target_size - new_h) // 2
        x_off = (target_size - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        metadata = {"original_size": (h, w), "scale": scale, "offset": (x_off, y_off)}
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), metadata

    def postprocess(self, predictions: Dict, metadata: Dict, conf_threshold: Optional[float] = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        boxes = predictions["boxes"].cpu().detach().numpy()
        scores = predictions["scores"].cpu().detach().numpy()
        labels = predictions["labels"].cpu().detach().numpy()
        scale = metadata.get("scale", 1.0)
        x_off, y_off = metadata.get("offset", (0, 0))
        orig_h, orig_w = metadata["original_size"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            x1 = max(0, min((box[0] - x_off) / scale, orig_w))
            y1 = max(0, min((box[1] - y_off) / scale, orig_h))
            x2 = max(0, min((box[2] - x_off) / scale, orig_w))
            y2 = max(0, min((box[3] - y_off) / scale, orig_h))
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": int(label),
                "class_name": self.categories.get(int(label), "Unknown"),
                "confidence": float(score),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            })
        return detections

    def predict(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        with torch.no_grad():
            tensor, metadata = self.preprocess(image)
            predictions = self.model([tensor.squeeze(0)])[0]
            return self.postprocess(predictions, metadata, conf_threshold)
