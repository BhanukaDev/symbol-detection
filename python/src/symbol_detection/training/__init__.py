"""Model training module"""
from symbol_detection.training.trainer import Trainer
from symbol_detection.training.losses import CIoULoss
from symbol_detection.training.data import COCODetectionDataset, collate_fn

__all__ = ['Trainer', 'CIoULoss', 'COCODetectionDataset', 'collate_fn']