"""Shared configuration for symbol detection"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
SYMBOLS_DIR = DATA_DIR / "electrical-symbols"
DISTRACTOR_DIR = DATA_DIR / "furnitures-and-other"

# Dataset settings
DATASET_SETTINGS = {
    "num_images": 10,
    "rows": (10, 60),
    "cols": (10, 60),
    "cell_size": (15, 25),
    "apply_symbol_effects": True,
    "apply_image_effects": True,
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    "show_labels": True,
    "bbox_thickness": 2,
    "font_scale": 0.5,
}
