# Weights & Biases (wandb) Integration Guide

## Overview
Weights & Biases has been integrated alongside TensorBoard to provide cloud-based experiment tracking and visualization for your event-based segmentation experiments.

## Installation

1. Install wandb (already added to requirements.txt):
```bash
pip install wandb
```

2. Login to your wandb account:
```bash
wandb login
```
You'll be prompted to enter your API key from https://wandb.ai/authorize

## Configuration

### In YAML Config Files
Both `model_config.yaml` and `model_config_event_scapes.yaml` now include wandb configuration:

```yaml
training:
  tensorboard_save_log: true  # Keep TensorBoard enabled
  wandb_save_log: false  # Set to true to enable W&B logging
  wandb_entity: null  # Your W&B team/entity name (username or team)
  wandb_project: "event-segmentation"  # Your W&B project name
  wandb_run_name: null  # Optional: custom run name (auto-generated if null)
```

### Enable wandb Logging
To enable wandb logging for your experiments, simply set:
```yaml
wandb_save_log: true
```

## What Gets Logged

### Epoch-Level Metrics (Every Epoch)
- **Training Loss**: `train_loss`
- **Validation Loss**: `val_loss`
- **Validation Mean IoU**: `val_mean_iou` (for segmentation models)
- **Learning Rate**: `learning_rate`

### Inference Metrics (When Enabled)
- **Inference Mean IoU**: `inference_mean_iou`
- **Inference Weighted Mean IoU**: `inference_weighted_mean_iou`
- **Inference Loss**: `inference_loss`

### Visualizations
- **Training Batches**: First 10 batches per epoch
  - RGB images
  - Event frames
  - Ground truth masks (colorized)
  - Predicted masks (colorized)
  - Combined grids showing all inputs side-by-side
- **Validation Batches**: First 10 batches during validation
  - Same visualization as training
- **Inference/Test Batches**: First 10 batches during inference (when enabled)
  - Same visualization as training and validation
  - Logged during `enable_inference_during_training` at specified intervals

### Model Metadata
- Full configuration
- Experiment timestamp
- Dataset information

**Note**: Gradient and parameter logging is disabled by default to save storage space. wandb is configured to match TensorBoard's logging behavior exactly - metrics and visualizations only.

## Features

### Both TensorBoard and wandb Run Simultaneously
- TensorBoard: Local visualization, immediate feedback
- wandb: Cloud-based, accessible from anywhere, better collaboration
- **Real-time logging**: Both systems update immediately during training

### Live Visualization During Training
wandb is configured to upload images immediately as they're logged:
- **Training images** (first 10 batches) upload as soon as they're processed
- **Validation images** (first 10 batches) upload during validation
- **Epoch metrics** upload at the end of each epoch
- All viewable in real-time on wandb.ai while training runs

### Automatic Resume
If training is interrupted and restarted with the same trial timestamp, wandb will automatically resume logging to the same run.

### Run Organization
- Runs are organized by project (configurable via `wandb_project`)
- Each run gets a unique name based on model and timestamp
- Custom run names can be set via `wandb_run_name`

## Usage Examples

### Basic Training with wandb
```yaml
# model_config.yaml
training:
  tensorboard_save_log: true
  wandb_save_log: true
  wandb_project: "event-segmentation"
```

```bash
python event_seg/train.py --config event_seg/model_config.yaml
```

### Custom Project and Run Name
```yaml
training:
  wandb_save_log: true
  wandb_entity: "your-team-name"  # Your W&B team or username
  wandb_project: "my-event-experiments"
  wandb_run_name: "smp_unet_resnet34_50ms"
```

### Using Team/Organization Account
```yaml
training:
  wandb_save_log: true
  wandb_entity: "eventresearch"  # Your team name on W&B
  wandb_project: "event-segmentation"
```

### Disable wandb (Use Only TensorBoard)
```yaml
training:
  tensorboard_save_log: true
  wandb_save_log: false
```

## Accessing Your Experiments

1. Go to https://wandb.ai
2. Navigate to your project (e.g., "event-segmentation")
3. View all runs, compare metrics, and analyze visualizations
4. Share experiment links with collaborators

## Benefits Over TensorBoard Alone

1. **Cloud Storage**: Access experiments from anywhere
2. **Collaboration**: Easy sharing with team members
3. **Comparison**: Better tools for comparing multiple runs
4. **Organization**: Projects, tags, and notes for experiment management
5. **Alerts**: Set up alerts for metric thresholds
6. **Reports**: Create shareable reports with results and visualizations
7. **Hyperparameter Tracking**: Automatic logging of all config parameters

## Tips

- Use descriptive `wandb_project` names for different experiment categories
- Set `wandb_run_name` for important experiments to make them easy to find
- Both TensorBoard and wandb can run simultaneously without performance impact
- wandb logs are persistent even if you delete local TensorBoard logs
- Use wandb's sweep feature for hyperparameter optimization (can be added later)

## Storage Optimization

wandb is configured to be storage-efficient and matches TensorBoard's logging behavior:

✅ **What is logged** (lightweight):
- Scalar metrics (loss, IoU, learning rate)
- Image visualizations (first 10 batches per epoch)
- Configuration parameters
- System metrics (CPU, GPU, memory)

❌ **What is NOT logged** (to save space):
- Model gradients (storage-heavy)
- Model parameters/weights at each step
- Full model checkpoints (use local storage instead)
- Activation histograms

This ensures wandb logs are efficient while providing all the visualization and metrics you need for experiment tracking.

## Troubleshooting

### Permission Denied Error (403)
If you see: `Error uploading run: returned error 403: permission denied`

This happens when:
1. The project doesn't exist in your W&B account
2. You need to re-authenticate

**Solution:**
```bash
# Re-login to wandb
wandb login --relogin

# Or create the project first on wandb.ai
# Go to: https://wandb.ai
# Click "New Project" and create a project with the name from your config
```

The training will continue with TensorBoard only if wandb fails to initialize. You'll see a warning message but training won't stop.

### Authentication Issues
```bash
wandb login --relogin
```

### Disable wandb Without Changing Code
Set environment variable:
```bash
export WANDB_MODE=disabled
python event_seg/train.py --config event_seg/model_config.yaml
```

### Offline Mode
If you want to log locally and sync later:
```bash
export WANDB_MODE=offline
python event_seg/train.py --config event_seg/model_config.yaml
# Later sync with: wandb sync
```

### Disable wandb in Config
Set in your YAML config:
```yaml
training:
  wandb_save_log: false
```
