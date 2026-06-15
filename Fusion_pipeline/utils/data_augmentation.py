import torch
import random
import numpy as np
"""
    Data augmentation functions.

    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor

    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).

    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""

class SharedSpatialAug:
    """Apply one shared random rotation (0,90,180,270°) + optional flips to all fields."""
    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, sample):
        # pick rotation multiple of 90°
        k = random.randint(0, 3)
        # pick flips
        do_h = random.random() < self.p_hflip
        do_v = random.random() < self.p_vflip

        def warp(x):
            # x: Tensor [C,H,W]
            x = torch.rot90(x, k, dims=(1,2))
            if do_h: x = x.flip(2)
            if do_v: x = x.flip(1)
            return x

        # warp every field in-place
        for key, val in sample.items():
            # only warp tensors
            if isinstance(val, torch.Tensor):
                sample[key] = warp(val)
        return sample


class NoiseAug:
    """Add Gaussian noise to only the specified input keys."""
    def __init__(self, std=0.01, keys=('power',)):
        self.std = std
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                x = sample[key]
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x)
                sample[key] = x + torch.randn_like(x) * self.std
        return sample


class ComposeSample:
    """Compose sample-level transforms (dict -> dict)."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    

