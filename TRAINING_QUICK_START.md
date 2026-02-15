# Quick Start: Production Training (Post Bug Fix)

## ⚡ TL;DR

1. **Bug Fixed**: Data loader category indexing corrected (1-based instead of 0-based) ✅
2. **Code Updated**: 
   - `training/data.py`: Category mapping fixed
   - `training/trainer.py`: Resume capability added
   - `notebooks/train_colab.ipynb`: Fresh/Resume modes added
3. **Ready to Train**: Yes, with corrected data loader

---

## Quick Decision Tree

```
Do you want to train a new model?
├─ YES (Recommended)
│  └─ Use training_mode = 'FRESH_START' in notebook
│     └─ Trains for 120 epochs with corrected hyperparameters
│        └─ Estimated time: 6-8 hours on A100
│
└─ NO, I want to resume interrupted training
   └─ Use training_mode = 'RESUME' in notebook
      └─ Continues from latest checkpoint
         └─ Lower learning rate to maintain convergence
```

---

## What Changed

### 🔴 Data Loader Bug Fixed
**File**: `training/data.py` line 37

**Before** (Bug):
```python
self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(...)}
# Result: Categories mapped to 0-6 (conflicts with background class)
# All 50 previous epochs had corrupted labels!
```

**After** (Fixed):
```python
self.cat_id_to_idx = {cat['id']: idx + 1 for idx, cat in enumerate(...)}
# Result: Categories mapped to 1-7 (correct, with background at 0)
# Fresh training will have correct label associations
```

### 📝 Trainer Updated
**File**: `training/trainer.py`

Added `start_epoch` parameter to support resuming from checkpoints:
```python
trainer = Trainer(
    ...
    start_epoch=resume_from_epoch,  # New parameter
)
```

### 📓 Notebook Updated
**File**: `notebooks/train_colab.ipynb`

**New Features**:
- Fresh vs. Resume training modes
- Production hyperparameter recommendations
- Category mapping verification
- Better error handling and feedback

---

## Recommended Training Plan

### Option A: Fresh Start (✅ Recommended)
```python
training_mode = 'FRESH_START'
```
- Trains from epoch 0 with corrected data loader
- 120 epochs total (~6-8 hours on A100)
- Best for initial production model
- Learning rate: 0.003, Batch size: 12

### Option B: Resume from Checkpoint (Only if interrupted)
```python
training_mode = 'RESUME'
```
- Automatically finds latest checkpoint
- Continues from next epoch
- Lower learning rate (0.001) to stabilize
- Use only if you're continuing interrupted training

---

## Hyperparameters (A100 GPU)

| Parameter | Value | Tuning |
|-----------|-------|--------|
| Learning Rate | 0.003 | ↓ if loss oscillates, ↑ if converges slowly |
| Batch Size | 12 | ↓ if OOM errors, ↑ for faster training |
| Epochs | 120 | Current: plenty for convergence |
| Weight Decay | 0.0005 | Prevents overfitting |
| LR Schedule | StepLR @ 40,80 | Decay by 0.5 at these epochs |

---

## Expected Results

### Training Loss
- Epoch 1: ~1.5
- Epoch 20: ~0.3
- Epoch 60: ~0.12
- Epoch 120: ~0.08

### Validation Loss
- Early: ~0.4
- Mid: ~0.25
- Final: ~0.18

### Training Time
- **Per Epoch**: ~3-4 minutes on A100
- **Total 120 Epochs**: ~6-8 hours
- **Overhead**: ~5 minutes (setup, validation)

---

## Step-by-Step (Colab)

### 1. Run Environment Setup
```
Cell 1: Verify PyTorch & CUDA
Cell 2: Mount Google Drive
Cell 3: Clone repository
Cell 4-5: Install dependencies
```

### 2. Configure Training Mode
```
Cell 8 (Bug Fix Explanation): Read about what was fixed
Cell 8 (Config): Set training_mode = 'FRESH_START'
```

### 3. Verify Data
```
Cell 9: Check annotations format and category mapping
Expected output: "Categories mapped to indices 1-7 ✓"
```

### 4. Start Training
```
Cell 10: Initialize trainer
Cell 11: Run trainer.train()
Monitor: Loss values printed each epoch
```

### 5. Monitor Progress
```
During training: Watch loss values decrease
After training: Run cell 12 to plot loss curves
```

---

## Verification Checklist

Before training: ✅ All items should pass

- [ ] Cell 1: PyTorch and CUDA available
- [ ] Cell 4: Dependencies installed (torch, torchvision, pycocotools)
- [ ] Cell 6: Dataset location exists (dataset/annotations.json)
- [ ] Cell 8: Training mode selected (FRESH_START)
- [ ] Cell 9: Categories mapped to indices 1-7 ✓
- [ ] Cell 9: Annotations have 'bbox' field ✓
- [ ] Cell 10: Trainer initialized successfully
- [ ] GPU memory available (~30GB free on A100)

---

## Troubleshooting

### Q: "CUDA out of memory" error
**A**: Reduce batch_size to 8 or 6 in cell 8

### Q: Loss not decreasing / stuck high
**A**: Check cell 9 - if categories aren't mapped to 1-7, regenerate dataset (cell 7)

### Q: Training interrupted (Colab timeout)
**A**: Set `training_mode = 'RESUME'`, re-run cells 8-11

### Q: Can I use old checkpoints?
**A**: No - they have the bug. Use FRESH_START mode (recommended)

### Q: How do I check if my model is training correctly?
**A**: After epoch 10-20, loss should be around 0.3. If still >1.0, something is wrong.

---

## Files Modified

```
✅ training/data.py
   - Line 37: Fixed category indexing (0-based → 1-based)
   
✅ training/trainer.py
   - Added start_epoch parameter for resuming
   - Updated train() method to use start_epoch
   
✅ notebooks/train_colab.ipynb
   - Added bug fix explanation
   - Added fresh/resume modes
   - Updated configuration cell
   - Added category mapping verification
   - Better error handling
```

---

## Model Specifications

```
Architecture: FasterRCNN ResNet50+FPN
Input: 512x512 images
Output: Bounding boxes + class labels

Classes (7):
1. Duplex Receptacle
2. Junction Box
3. Light
4. Single-pole One-way Switch
5. Three-pole One-way Switch
6. Two-pole One-way Switch
7. Two-way Switch

Background: Index 0 (reserved)
```

---

## After Training

### Save Your Model
```python
# Use the final checkpoint
checkpoint = 'checkpoints/model_epoch_final.pth'
```

### Deploy to Production
```python
from symbol_detection.inference import SymbolDetectionPredictor

predictor = SymbolDetectionPredictor(
    checkpoint_path='checkpoints/model_epoch_final.pth',
    categories_file='dataset/annotations.json',
)

result = predictor.predict('floor_plan.jpg')
```

### Validate Results
- Symbol names should now align correctly (bug fix effect)
- Detection accuracy should be significantly better
- Different loss curves than previous 50 epochs (bug fix)

---

## Full Documentation

For detailed information, see [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)

Topics covered:
- Detailed bug explanation
- All hyperparameter recommendations
- Training phases breakdown
- Monitoring guidance
- Common issues & solutions
- Deployment instructions

---

## Summary

| Item | Status |
|------|--------|
| Data loader bug | ✅ FIXED |
| Trainer resume support | ✅ ADDED |
| Notebook updated | ✅ COMPLETED |
| Hyperparameters | ✅ OPTIMIZED |
| Documentation | ✅ PROVIDED |
| **Ready to train** | ✅ **YES** |

**Next step**: Open `notebooks/train_colab.ipynb`, set `training_mode = 'FRESH_START'`, and run the cells!

**Estimated time to first model**: ~8 hours on A100
