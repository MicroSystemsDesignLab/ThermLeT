# test_single_plot.py

import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import build_temperature_model
from metrics import eval_metrics

def parse_args():
    parser = argparse.ArgumentParser("Test & plot single sample")
    parser.add_argument('--inp',       type=str, required=True,
                        help="Path to input .npy (H,W,5)")
    parser.add_argument('--target',    type=str, required=True,
                        help="Path to true temperature .npy (H,W)")
    parser.add_argument('--model',     type=str, required=True,
                        help="Path to your checkpoint (best.pth etc.)")
    parser.add_argument('--out_dir',   type=str, default='.',
                        help="Directory to save outputs")
    parser.add_argument('--heat_value',type=float, default=950.0,
                        help="Override channel index=2 with this value")
    parser.add_argument('--tol',       type=float, default=1.0,
                        help="Tolerance for accuracy metric")
    parser.add_argument('--device',    type=str, default='cuda',
                        help="cpu or cuda")
    # model‐config must match training
    parser.add_argument('--img_size', nargs=2, type=int,   default=[224,224])
    parser.add_argument('--patch_size',        type=int,   default=16)
    parser.add_argument('--hidden_dim',        type=int,   default=768)
    parser.add_argument('--num_layers',        type=int,   default=12)
    parser.add_argument('--nheads',            type=int,   default=12)
    parser.add_argument('--dim_feedforward',   type=int,   default=2048)
    parser.add_argument('--mlp_ratio',         type=float, default=4.)
    parser.add_argument('--qkv_bias',          action='store_true')
    parser.add_argument('--dropout',           type=float, default=0.1)
    parser.add_argument('--num_res_blocks',    type=int,   default=1)
    parser.add_argument('--filters',           type=int,   default=64)
    return parser.parse_args()

def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- 1) Build & load model ---
    model = build_temperature_model(args)
    ckpt  = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # --- 2) Load input and ground truth ---
    inp = np.load(args.inp)            # (H, W, 5)
    inp[...,2] = args.heat_value       # override heat channel
    power   = inp[...,0].astype(np.float32)         # (H, W)
    coords  = inp[...,3:5].astype(np.float32)       # (H, W, 2)
    coords  = np.transpose(coords, (2,0,1))         # (2, H, W)
    gt_temp = np.load(args.target).astype(np.float32)  # (H, W)
    print("gt_temp",gt_temp.shape)
    # --- 3) To torch tensors + batch dim ---
    p_t = torch.from_numpy(power)[None,None].to(device)   # [1,1,H,W]
    c_t = torch.from_numpy(coords)[None,...].to(device)   # [1,2,H,W]

    # --- 4) Inference ---
    with torch.no_grad():
        out = model(p_t, c_t)   # [1,1,H,W]
    pred = out.cpu().numpy()[0,0]  # (H, W)

    # --- 5) Compute metrics ---
    mae  = np.mean(np.abs(pred - gt_temp[:,:,0]))
    rmse = np.sqrt(np.mean((pred - gt_temp[:,:,0])**2))
    # R^2
    ss_res = np.sum((pred - gt_temp[:,:,0])**2)
    ss_tot = np.sum((gt_temp[:,:,0] - gt_temp[:,:,0].mean())**2)
    r2     = 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
    # tolerance‐based accuracy
    acc    = np.mean(np.abs(pred - gt_temp[:,:,0]) < args.tol)

    print(f"MAE = {mae:.4f}   RMSE = {rmse:.4f}")
    print(f"R²  = {r2:.4f}   Acc(±{args.tol}) = {acc*100:.2f}%")

    # --- 6) Save raw prediction array ---
    base      = os.path.splitext(os.path.basename(args.inp))[0]
    npy_path  = os.path.join(args.out_dir, f"{base}_pred.npy")
    np.save(npy_path, pred)
    print(f"Saved prediction array → {npy_path}")

    # --- 7) Plot Input / GT / Pred side by side ---
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    im0 = axes[0].imshow(power,    cmap='magma')
    axes[0].set_title('Input Power');    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gt_temp[:,:,0],  cmap='hot')
    axes[1].set_title('Ground Truth Temp'); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred,     cmap='hot')
    axes[2].set_title('Predicted Temp');    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"{base} — MAE:{mae:.3f}, RMSE:{rmse:.3f}, R²:{r2:.3f}, Acc±{args.tol}:{acc*100:.1f}%")
    plt.tight_layout(rect=[0,0,1,0.95])

    img_path = os.path.join(args.out_dir, f"{base}_comparison.png")
    plt.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot → {img_path}")

if __name__ == "__main__":
    main()
