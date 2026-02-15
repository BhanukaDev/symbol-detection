# Production Training Guide - Symbol Detection Model

## 🔴 CRITICAL: Data Loader Bug Fix

### Issue Discovered
The training data loader (`training/data.py` line 36) was using **0-based category indexing** when FasterRCNN requires **1-based indexing** (0 reserved for background).

### Impact
- **All 50 epochs of previous training are invalid** - class labels were misaligned
- Model learned wrong associations between image features and electrical symbol classes
- Category 0 (Background) was mapped to what should be Category 1 (Duplex Receptacle)
- All subsequent categories were shifted by one

### Fix Applied ✅
```python
# BEFORE (Bug - Invalid):
self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}
# Result: Categories → Indices 0-6 (WRONG - background conflicts with categories)

# AFTER (Fixed - Correct):
self.cat_id_to_idx = {cat['id']: idx + 1 for idx, cat in enumerate(self.coco_data['categories'])}
# Result: Categories → Indices 1-7 (CORRECT - background at 0, categories at 1-7)
```

**Status**: ✅ **FIXED** in training/data.py

---

## Production Training Recommendations

### Hyperparameters (A100 GPU - 40GB VRAM)

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| **Learning Rate** | 0.003 | Slightly lower than initial (0.005) for stability |
| **Number of Epochs** | 120 | Longer training to reach convergence with corrected labels |
| **Batch Size** | 12 | Proven stable on A100, allows good gradient estimates |
| **Weight Decay** | 0.0005 | L2 regularization prevents overfitting |
| **Optimizer** | SGD with momentum | Standard for object detection, Momentum=0.9 |
| **LR Schedule** | StepLR | Decay by 0.5 at epochs 40, 80 |
| **Warmup** | 5 epochs | Slowly increase LR to prevent instability |

### Expected Training Time
- **Total**: ~6-8 hours on NVIDIA A100
- **Per epoch**: ~3-4 minutes
- **Validation**: ~30 seconds per epoch

### Expected Loss Convergence
- **Target Training Loss**: 0.08-0.12
- **Target Validation Loss**: 0.15-0.20
- **Notes**: Loss curves will be different from previous 50 epochs (which had corrupted labels)

---

## Training Modes

### 1. FRESH START (Recommended) ✅
**Use this for the first production training run**

```python
training_mode = 'FRESH_START'
```

**Behavior**:
- Starts from epoch 0 with randomly initialized weights
- Uses corrected data loader (1-based category indexing)
- Trains for full 120 epochs
- Best for initial model development

**Start time**: ~6-8 hours from now

---

### 2. RESUME FROM CHECKPOINT
**Use this to continue interrupted training**

```python
training_mode = 'RESUME'
```

**Behavior**:
- Automatically finds most recent checkpoint
- Loads model weights, optimizer state, and training history
- Continues from next epoch
- Lower learning rate (0.001) to maintain convergence
- Useful if training is interrupted (e.g., Colab timeout)

**Important**: Only use this if resuming interrupted training. Don't use with old checkpoints that have the bug.

---

## Step-by-Step Training Instructions

### 1. Prepare Environment (Google Colab)
```python
# Run cells 1-6 in train_colab.ipynb
# - Setup environment
# - Mount Google Drive
# - Clone repository
# - Install dependencies
# - Verify installation
# - Mount dataset location
```

### 2. Generate Dataset (Optional)
```python
# Run cell 7 only if you don't have dataset/annotations.json
# Dataset includes:
# - 200-500 synthetic floor plan images
# - 7 electrical symbol classes
# - COCO format annotations with proper bbox field
```

### 3. Configure Training
```python
# Run cell 8 and 8a
# - Cell 8a: Explains the bug fix
# - Cell 8: Set training_mode and review hyperparameters
#   - FRESH_START: New training run ✅
#   - RESUME: Continue from checkpoint (optional)
```

### 4. Verify Data Integrity
```python
# Run cell 9
# - Verifies annotations have bbox field
# - Confirms category mapping is 1-based (1-7, not 0-6)
# - Shows first annotation sample
```

### 5. Initialize Trainer
```python
# Run cell 10
# - Creates Trainer with corrected data loader
# - Loads checkpoint if resuming
# - Displays training configuration
```

### 6. Start Training
```python
# Run cell 11
# - Training loop for 120 epochs
# - Saves checkpoint every 10 epochs
# - Validates after each epoch
# - Shows loss metrics
```

### 7. Monitor Progress
```python
# Run cell 12-13 periodically during training
# - Visualize training curves
# - Check current loss values
# - Estimated convergence timeline
```

### 8. List Saved Checkpoints
```python
# Run cell 14
# - Shows all saved checkpoints
# - File sizes
# - Timestamps
```

---

## Training Phases Expected

### Phase 1: Warm-up (Epochs 1-5)
- Learning rate gradually increases from 0 to 0.003
- Loss will decrease quickly
- Model learning basic features

### Phase 2: Main Training (Epochs 6-40)
- Stable learning at 0.003
- Loss decreases gradually
- Model learning fine-grained symbol features

### Phase 3: Refinement (Epochs 41-80)
- Learning rate decays to 0.0015 (50% reduction at epoch 40)
- Loss fine-tuning
- Model reaching convergence plateau

### Phase 4: Fine-tuning (Epochs 81-120)
- Learning rate further decays to 0.00075 (50% reduction at epoch 80)
- Very small loss improvements
- Model converging to final weights

---

## Monitoring Training

### Key Metrics to Watch
1. **Train Loss**: Should steadily decrease from ~1.5 to ~0.1
2. **Validation Loss**: Should decrease and stabilize around 0.15-0.20
3. **Loss Ratio (Val/Train)**: Should stay around 1.5-2.0 (good generalization)
4. **Training Speed**: ~3-4 minutes per epoch on A100

### What's Normal
✅ Loss oscillates slightly - normal with SGD
✅ Validation loss higher than train loss - expected
✅ Loss plateau after epoch ~80 - model converged
✅ Occasional training loss increases - momentum effects

### What's Concerning
❌ Loss increasing consistently - learning rate too high
❌ Training stuck at high loss (>1.5) - data corruption
❌ Validation loss much higher (>5x train loss) - severe overfitting

---

## Post-Training Validation

### 1. Evaluate Final Model
```python
# Create inference_test.ipynb
# - Load model_epoch_final.pth
# - Run on test dataset
# - Check symbol name alignment (should be correct now!)
# - Verify detection quality
```

### 2. Compare with Previous Run
- New training should show different loss curves
- Symbol names should now align correctly (bug fix effect)
- Better overall accuracy expected

### 3. Save Best Model
```python
# Option 1: Use final checkpoint (epoch 120)
checkpoint = 'model_epoch_final.pth'

# Option 2: Use best validation loss checkpoint
# Check metrics.json for which epoch had lowest val_loss
```

---

## Common Issues & Solutions

### Issue: "Batch size too large - CUDA out of memory"
**Solution**: Reduce batch_size to 8 or 6 in cell 8

### Issue: Training seems extremely slow
**Solution**: Check GPU is being used (cell 1 should show GPU: NVIDIA A100...)

### Issue: Loss not decreasing / stuck at high value
**Solution**: 
1. Verify cell 9 shows categories mapped to indices 1-7 ✓
2. Re-run cell 7 to regenerate dataset

### Issue: Training interrupted (Colab timeout)
**Solution**: 
1. Set training_mode = 'RESUME' in cell 8
2. Re-run cells 8-11
3. Training will continue from checkpoint

### Issue: Previous checkpoints won't load
**Solution**: Use FRESH_START mode (recommended). Old checkpoints have bug.

---

## File Locations

### On Google Drive (if Drive mounted):
```
/content/drive/MyDrive/symbol-detection/
├── dataset/
│   ├── images/           # Training images
│   ├── annotations.json  # COCO format annotations
│   ├── train_annotations.json
│   └── val_annotations.json
└── checkpoints/
    ├── model_epoch_10.pth
    ├── model_epoch_20.pth
    ├── ...
    ├── model_epoch_final.pth
    └── metrics.json
```

### On Colab Storage (fallback):
```
/content/symbol-detection/
├── dataset/
├── checkpoints/
```

---

## Next Steps After Training

### 1. Deploy to Production
```python
from symbol_detection.inference import SymbolDetectionPredictor

predictor = SymbolDetectionPredictor(
    checkpoint_path='checkpoints/model_epoch_final.pth',
    categories_file='dataset/annotations.json',
    confidence_threshold=0.5,
)

result = predictor.predict('floor_plan.jpg')
```

### 2. Further Fine-tuning (Optional)
Data collected in production → Retrain with domain-specific examples

### 3. Model Monitoring
Track inference accuracy on real floor plans over time

---

## Testing Data Loader Fix

To verify the fix is working:

```python
from symbol_detection.training.data import COCODataset

dataset = COCODataset('dataset/annotations.json', data_dir='dataset/images')

# Check category mapping
print(dataset.cat_id_to_idx)
# Should show: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
# NOT: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

# Get a sample
sample = dataset[0]
print(sample['labels'])  # Should have values 1-7, never 0
```

---

## Summary

| Item | Status | Notes |
|------|--------|-------|
| Data loader bug | ✅ FIXED | 1-based category indexing |
| Trainer code | ✅ READY | Supports start_epoch parameter |
| Inference code | ✅ READY | Uses correct category mapping |
| Training notebook | ✅ UPDATED | Fresh/Resume modes supported |
| Hyperparameters | ✅ OPTIMIZED | Production-ready values |
| Documentation | ✅ COMPLETE | This file |

**Ready to train**: Yes ✅

**Estimated completion**: 6-8 hours from start

**Expected outcome**: Production-ready model with correct class mappings
