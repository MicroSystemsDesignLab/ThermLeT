# U-Net Baseline

This folder contains the CNN/U-Net baseline used to compare against the ThermLeT transformer pipeline.

The baseline treats thermal prediction as an image-to-image regression task: an input power map is passed through an encoder-decoder network to predict the steady-state temperature map.

## Files

```text
U-Net_Baseline/
├── main.py
├── unet_final.py
├── unet_multigpu_final.py
├── UNet_final.ipynb
└── UNet_multigpu_final.ipynb
```

The `.py` files are exported from Colab notebooks and include dataset loading, model definition, training, evaluation, and visualization code.

## Input format

The baseline uses power maps and temperature maps stored as `.npy` arrays. The common expected shapes are:

```text
input power map      : H × W × C
training input used  : H × W × 1
temperature target   : H × W × 1
```

The scripts generally select the first input channel:

```python
X = power_maps[..., 0:1]
Y = temp_maps[..., 0:1]
```

## Model summary

The U-Net model consists of:

- convolutional encoder blocks,
- max-pooling downsampling,
- a convolutional bottleneck,
- transposed-convolution upsampling,
- skip connections, and
- a final `1×1` convolution for temperature regression.

The output activation is linear because temperature prediction is a continuous regression task.

## Typical workflow

1. Mount or provide access to the dataset.
2. Load `.npy` input and target arrays.
3. Resize samples to `224 × 224`.
4. Split into train/test sets.
5. Train the U-Net with MSE loss and MAE as a metric.
6. Save the best model checkpoint.
7. Evaluate MAE, latency, throughput, and approximate FLOPs.
8. Visualize input power, ground-truth temperature, and predicted temperature.

## Notes

- Some scripts contain Colab-specific paths such as `/content/drive/MyDrive/...`; update these paths before running locally.
- This baseline uses only the power-map channel, unlike the ThermLeT fusion pipeline, which also uses coordinate-aware spatial information.
- The baseline is intended for comparison and ablation, not as the primary ThermLeT model.
