"""Dataset generation module"""
from symbol_detection.dataset.generator import COCODatasetGenerator, generate_coco_dataset
from symbol_detection.dataset.cli import main

__all__ = [
    "COCODatasetGenerator",
    "generate_coco_dataset",
    "main",
]
