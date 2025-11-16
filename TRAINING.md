# Chess Model Training Guide

## Overview
The chess model has been optimized with persistence and better training parameters.

## Changes Made

### Training Optimizations
- **Epochs**: Increased from 5 to 15 for better convergence
- **Learning Rate Scheduler**: ReduceLROnPlateau automatically reduces LR when loss plateaus
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Checkpointing**: Saves best model and periodic checkpoints every 5 epochs

### Persistence
Models are now saved to a **Modal Volume** (`chess-models`) so they persist after training.

## Usage

### 1. Train the Model

```bash
modal run src/model.py::run_training
```

This will:
- Train for 15 epochs on 100k positions
- Save checkpoints to Modal Volume `chess-models`:
  - `best_model.pt` - Best model by loss
  - `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, etc. - Periodic checkpoints
  - `final_model.pt` - Final model after all epochs

### 2. List Available Checkpoints

```bash
modal run src/model.py::list_checkpoints
```

### 3. Download a Trained Model

```python
from src.model import app

# Download best model
with app.run():
    model_bytes = app.download_model.remote("best_model.pt")
    
    # Save locally
    with open("best_model.pt", "wb") as f:
        f.write(model_bytes)
```

### 4. Load a Model for Inference

```python
from src.model import load_model

# Load from local file
model = load_model("best_model.pt")

# Use for predictions
policy_logits, value_pred = model(board_tensor)
```

## Saved Checkpoint Format

Each checkpoint contains:
```python
{
    'epoch': int,                      # Epoch number
    'model_state_dict': OrderedDict,   # Model weights
    'optimizer_state_dict': OrderedDict, # Optimizer state
    'loss': float                      # Loss at this checkpoint
}
```

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 15 | Increased from 5 |
| Batch Size | 128 | |
| Learning Rate | 1e-3 | Initial, reduced by scheduler |
| Dataset Size | 100k | Filtered from 200k |
| Optimizer | Adam | |
| LR Scheduler | ReduceLROnPlateau | factor=0.5, patience=2 |
| Gradient Clipping | 1.0 | max_norm |
| GPU | T4 | Requested in Modal |

## Next Steps

If you want to train longer:
- Increase `NUM_EPOCHS` in `src/model.py` (currently 15)
- Consider increasing dataset size beyond 100k
- Experiment with different learning rates

## Troubleshooting

**Volume not found?**
```bash
modal volume create chess-models
```

**Out of memory?**
- Reduce batch_size from 128 to 64
- Use smaller GPU (though T4 is already small)

**Model not improving?**
- Check if learning rate is too high/low
- Increase epochs
- Add more data

