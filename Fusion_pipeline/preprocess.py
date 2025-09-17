import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class ThermalDataset(Dataset):
    """
    PyTorch Dataset for thermal estimation.
    Only uses:
      - power       : inp[..., 0]
      - coords (x,y): inp[..., 3:5]
    Predicts:
      - temperature : out[..., 0] or out
    """
    def __init__(self, inp_dir, out_dir, transform=None):
        self.inp_paths = sorted(glob.glob(os.path.join(inp_dir, '*.npy')))
        self.out_paths = sorted(glob.glob(os.path.join(out_dir, '*.npy')))
        assert len(self.inp_paths) == len(self.out_paths), \
            f"Mismatch: {len(self.inp_paths)} inputs vs {len(self.out_paths)} outputs"
        self.transform = transform

    def __len__(self):
        return len(self.inp_paths)

    def __getitem__(self, idx):
        # load numpy arrays
        inp = np.load(self.inp_paths[idx])        # (H, W, 5)
        out = np.load(self.out_paths[idx])        # (H, W) or (H, W, C)

        # extract power [H,W]
        power = inp[..., 0].astype(np.float32)

        # extract coords [H,W,2] → [2,H,W]
        coords = inp[..., 3:5].astype(np.float32)
        coords = np.transpose(coords, (2, 0, 1))

        # process target temperature [H,W]
        if out.ndim == 3 and out.shape[2] > 1:
            temp = out[..., 0].astype(np.float32)
        else:
            temp = out.astype(np.float32)

        # convert to tensors, channel‐first
        sample = {
            'power':       torch.from_numpy(power)[None],    # [1,H,W]
            'coords':      torch.from_numpy(coords),         # [2,H,W]
            'temperature': torch.from_numpy(temp)[None],     # [1,H,W]
        }

        # apply any augmentation / preprocessing transforms
        if self.transform:
            sample = self.transform(sample)

        return sample



def get_train_val_loaders(
    inp_dir,
    out_dir,
    batch_size=16,
    num_workers=4,
    train_ratio=0.8,
    seed=42,
    step=1,
    transform=None
):
    """
    Returns train & validation DataLoaders over ThermalDataset(power+coords → temperature).
    """
    # build & (optionally) subsample dataset
    dataset = ThermalDataset(inp_dir, out_dir, transform=transform)
    if step > 1:
        indices = list(range(0, len(dataset), step))
        if not indices:
            raise ValueError(f"No data left after step={step} (dataset size={len(dataset)})")
        dataset = Subset(dataset, indices)

    # split
    total = len(dataset)
    if total == 0:
        raise ValueError(f"No samples found in {inp_dir} / {out_dir}")
    n_train = int(total * train_ratio)
    n_val = total - n_train
    if n_train == 0 or n_val == 0:
        raise ValueError(f"Train/val split invalid: total={total}, train={n_train}, val={n_val}")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# Example usage:
# transform = None  # or your ApplyToTensorFields(...)
# train_loader, val_loader = get_train_val_loaders(
#     inp_dir='generated_input',
#     out_dir='generated_target',
#     batch_size=8,
#     num_workers=4,
#     train_ratio=0.8,
#     seed=42,
#     step=1,
#     transform=transform
# )
