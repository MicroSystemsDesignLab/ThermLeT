# ThermLeT: Transformer-Based Temperature Prediction for 2.5D Chiplet Architectures

ThermLeT is a transformer-based surrogate model for fast steady-state thermal prediction in heterogeneous 2.5D chiplet packages. The goal of this repository is to support rapid thermal design-space exploration by replacing expensive finite-element thermal simulations with a learned model that predicts chip/package temperature maps from layout-aware inputs.

This repository accompanies the paper:

> **ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture**  
> Varun Darshana Parekh*, Anusha Devulapally*, Sivani Devarapalli, Cassius Henderson, Shimeng Yu, and Vijaykrishnan Narayanan
> (* co-first authorship)

## Why ThermLeT?

2.5D chiplet packages offer high bandwidth, modular integration, and better design flexibility, but dense chiplet placement also increases thermal risk. Traditional finite-element methods such as ANSYS provide high-fidelity results but are too slow for iterative floorplanning and design-space exploration.

ThermLeT learns a fast surrogate for chiplet thermal analysis by combining:

- power-density maps,
- spatial coordinate information,
- material/cooling context from the thermal stack, and
- transformer-based long-range spatial modeling.

In the paper, ThermLeT is trained on **6,221 ANSYS-generated chiplet layouts** from representative 2.5D architectures and achieves **3.16°C RMSE** and **2.24°C MAE** on the held-out test set, with approximately **13 ms/sample** inference latency.

## Repository structure

```text
ThermLeT-main/
├── Fusion_pipeline/          # Main transformer-based thermal prediction pipeline
│   ├── main.py               # Training script
│   ├── test.py               # Single-sample inference and visualization
│   ├── model.py              # ThermLeT transformer model implementation
│   ├── preprocess.py         # Dataset and dataloader utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── misc.py               # Losses and helper functions
│   ├── submodule.py          # Transformer and CNN building blocks
│   ├── requirements.txt      # Conda environment specification
│   └── utils/                # Path and augmentation utilities
│
├── U-Net_Baseline/           # CNN/U-Net baseline implementation
│   ├── main.py
│   ├── unet_final.py
│   ├── unet_multigpu_final.py
│   ├── UNet_final.ipynb
│   └── UNet_multigpu_final.ipynb
│
└── README.md
```

## Dataset

The dataset consists of generated input tensors and target temperature maps from finite-element simulations of 2.5D chiplet packages. The paper uses layouts perturbed from three representative architectures:

- Micro150,
- Ascend910,
- MultiGPU.

Each input sample is expected as a NumPy array with shape:

```text
H × W × 5
```

The current fusion pipeline uses:

```text
channel 0     : power-density map
channels 3-4  : normalized x/y coordinate maps
```

The target output is a temperature map stored as either:

```text
H × W
```

or

```text
H × W × C
```

where the first channel is used as the steady-state temperature target.

Dataset link:

[Download dataset](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/vdp5074_psu_edu/IgAC7cdyMKPzTZkzhtCJCGkuAY9TcDpKpl_7e6-bCd3p5IY?e=FxxCVq)

## Quick start

Clone the repository:

```bash
git clone https://github.com/MicroSystemsDesignLab/ThermLeT.git
cd ThermLeT
```

Create the environment:

```bash
cd Fusion_pipeline
conda create --name thermlet --file requirements.txt
conda activate thermlet
```

Train ThermLeT:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --inp_dir /path/to/generated_inputs \
  --out_dir /path/to/generated_targets \
  --batch_size 16 \
  --epochs 30
```

Run inference on one sample:

```bash
python test.py \
  --inp /path/to/sample_input.npy \
  --target /path/to/sample_target.npy \
  --model experiment/exp_1/checkpoints/best.pth \
  --out_dir results
```

The inference script saves:

- the predicted temperature map as `.npy`, and
- a side-by-side comparison plot of input power, ground-truth temperature, and predicted temperature.

## Outputs

Training artifacts are saved under:

```text
Fusion_pipeline/experiment/exp_1/
├── logs/                   # Training logs
├── tensorboard/            # TensorBoard summaries
├── checkpoints/            # Per-epoch checkpoints and best.pth
├── intermediate_results/   # Validation visualizations per epoch
├── loss_curve.png
└── error_curve.png
```

## Baselines

The repository includes a U-Net/CNN baseline under `U-Net_Baseline/`. This baseline is useful for comparing transformer-based fusion against an image-to-image convolutional temperature predictor.

## Notes on the current implementation

The paper-level ThermLeT formulation uses multimodal information including power maps, coordinates, material properties, and cooling/heat-transfer parameters. The current `Fusion_pipeline` implementation focuses on the power-map + coordinate-fusion transformer path. If extending the released code to match the complete paper configuration, add the scalar material/heat-transfer embedding branch described in the paper and connect it to the token-fusion stage.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{parekh2026thermlet,
  title     = {ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture},
  author    = {Parekh, Varun Darshana and Devulapally, Anusha and Devarapalli, Sivani and Henderson, Cassius and Yu, Shimeng and Narayanan, Vijaykrishnan},
  booktitle = {2026 39th International Conference on VLSI Design and 25th International Conference on Embedded Systems (VLSID)},
  year      = {2026}
}
```

## Acknowledgement

This work was supported by PRISM, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation program sponsored by DARPA.
