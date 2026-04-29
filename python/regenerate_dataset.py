import sys
from pathlib import Path

here = Path(__file__).parent
for p in ['src', 'floor-grid/src', 'effects/src']:
    sys.path.insert(0, str(here / p))

from symbol_detection.dataset.generator import COCODatasetGenerator

g = COCODatasetGenerator(
    output_dir='dataset',
    symbols_dir='data/electrical-symbols',
    distractor_dir='data/furnitures-and-other',
)
g.generate_dataset(num_images=10, rows=(20, 70), cols=(20, 70), cell_size=(8, 30))
g.save_annotations()
print('Done')
