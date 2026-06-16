# ThermLeT Fusion Pipeline

This folder contains the main transformer-based thermal prediction pipeline for ThermLeT. The model predicts steady-state temperature maps for 2.5D chiplet packages from layout-aware thermal inputs.

The implementation follows a fusion-style design: power-map tokens and coordinate-aware spatial tokens are embedded separately, fused through transformer encoder blocks, folded back into spatial feature maps, and decoded into a temperature field.

## Files

```text
Fusion_pipeline/
├── main.py                 # Training entry point
├── test.py                 # Single-sample inference and visualization
├── model.py                # TemperatureTransformer model
├── preprocess.py           # ThermalDataset and train/validation dataloaders
├── metrics.py              # MSE, MAE-like error, RMSE, and auxiliary metrics
├── misc.py                 # Multi-scale gradient loss and helper functions
├── submodule.py            # Patch embedding, attention, transformer blocks, RRDB
├── requirements.txt        # Conda environment file
└── utils/
    ├── data_augmentation.py
    └── path_utils.py
```

## Input and output format

The dataloader expects two directories:

```text
--inp_dir   directory containing input .npy files
--out_dir   directory containing target .npy files
```

The number of `.npy` files in both directories must match. Files are sorted alphabetically and paired by order, so use consistent naming.

### Input tensor

Each input file should be a NumPy array with shape:

```text
H × W × 5
```

The current code uses:

```text
inp[..., 0]    -> power-density map
inp[..., 3:5]  -> x/y coordinate maps
```

The model internally converts these to channel-first tensors:

```text
power   : [1, H, W]
coords  : [2, H, W]
```

### Target tensor

Each target file should be a NumPy array with shape:

```text
H × W
```

or

```text
H × W × C
```

If the target has multiple channels, the current code uses the first channel as the temperature target.

## Environment setup

Create and activate the conda environment:

```bash
conda create --name thermlet --file requirements.txt
conda activate thermlet
```

If you prefer a lighter custom environment, the core packages are:

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib kornia tensorboard
```

Use CUDA-enabled PyTorch for GPU training.

## Training

Run training from inside this folder:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --inp_dir /path/to/generated_inputs \
  --out_dir /path/to/generated_targets \
  --batch_size 16 \
  --epochs 30 \
  --lr 3e-4 \
  --weight_decay 1e-4
```

Common options:

```text
--batch_size       batch size, default 16
--epochs           number of training epochs, default 30
--img_size         input resolution, default 224 224
--patch_size       patch size for tokenization, default 16
--hidden_dim       transformer embedding dimension, default 768
--num_layers       transformer depth, default 12
--nheads           number of attention heads, default 12
--filters          convolutional feature width, default 64
--train_ratio      train/validation split, default 0.8
--step             subsample every Nth sample, useful for debugging
```

## Training outputs

The training script creates the following directory:

```text
experiment/exp_1/
├── logs/
├── tensorboard/
├── checkpoints/
├── intermediate_results/
├── loss_curve.png
└── error_curve.png
```

`checkpoints/best.pth` stores the checkpoint with the lowest validation loss.

To monitor training with TensorBoard:

```bash
tensorboard --logdir experiment/exp_1/tensorboard
```

## Single-sample inference

After training, run:

```bash
python test.py \
  --inp /path/to/sample_input.npy \
  --target /path/to/sample_target.npy \
  --model experiment/exp_1/checkpoints/best.pth \
  --out_dir results
```
Download pre-trained weights: https://github.com/MicroSystemsDesignLab/ThermLeT/releases/tag/model_weights


This produces:

```text
results/<sample_name>_pred.npy
results/<sample_name>_comparison.png
```

The comparison figure shows:

1. input power map,
2. ground-truth temperature map,
3. predicted temperature map.

## Model overview

The model in `model.py` contains four main stages:

1. **Power branch**: embeds the power-density map using patch embedding.
2. **Coordinate branch**: applies CoordConv to normalized x/y coordinate maps and embeds the resulting spatial features.
3. **Transformer encoder**: concatenates power and coordinate tokens, adds positional embeddings, and models long-range thermal dependencies.
4. **Spatial decoder**: folds tokens back to feature maps, fuses modality-specific features, refines them with residual dense blocks, and predicts the temperature map.

## Loss function

Training uses a weighted combination of:

```text
0.5 × L1 loss + 0.25 × multi-scale gradient loss
```

The L1 term improves temperature accuracy, while the gradient term encourages spatial consistency in the predicted temperature field.

## Notes

- Run scripts from inside `Fusion_pipeline/` so relative imports resolve correctly.
- Input and target filenames should be sorted in the same order.
- The current implementation consumes power and coordinate channels. To reproduce the full paper-level multimodal setup, extend the model with the material-property and heat-transfer scalar embedding branch.
