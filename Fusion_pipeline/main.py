# main.py

import os
import time
import argparse
import io
import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.path_utils import ensure_dir
from preprocess import get_train_val_loaders
from model import build_temperature_model
from metrics import *
from misc import *
from utils.data_augmentation import *

# ———————————————————
# Setup directories
# ———————————————————
BASE_DIR          = "experiment/exp_1"
LOG_DIR           = os.path.join(BASE_DIR, "logs")
TB_DIR            = os.path.join(BASE_DIR, "tensorboard")
CKPT_DIR          = os.path.join(BASE_DIR, "checkpoints")
INTERMEDIATE_DIR  = os.path.join(BASE_DIR, "intermediate_results")

for d in (LOG_DIR, TB_DIR, CKPT_DIR, INTERMEDIATE_DIR):
    ensure_dir(d)

# ———————————————————
# Configure logging
# ———————————————————
log_file = os.path.join(LOG_DIR, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_model_size(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024*1024)

def get_args_parser():
    parser = argparse.ArgumentParser('TempTransformer', add_help=False)
    # Data & loader
    parser.add_argument('--inp_dir',    type=str, default='processed_inputs')
    parser.add_argument('--out_dir',    type=str, default='processed_outputs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--train_ratio',type=float, default=0.8)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--step',       type=int,   default=1)
    # Augmentation
    parser.add_argument('--flip_prob',  type=float, default=0.5)
    # Model/hyperparams
    parser.add_argument('--img_size', nargs=2, type=int, default=[224,224])
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
    # Training
    parser.add_argument('--device',            type=str,   default='cuda')
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--epochs',            type=int,   default=30)
    parser.add_argument('--log_step',          type=int,   default=5)
    return parser


def main(args):
    # — Device & TensorBoard —
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}, GPUs: {torch.cuda.device_count()}")
    logging.info(f"Args: {args}")
    writer = SummaryWriter(log_dir=TB_DIR)

    spatial_tf = SharedSpatialAug(p_hflip=args.flip_prob, p_vflip=args.flip_prob)
    noise_tf   = NoiseAug(std=0.01, keys=('power',)) 
    full_transform = ComposeSample([spatial_tf, noise_tf])

    train_loader, val_loader = get_train_val_loaders(
        inp_dir=args.inp_dir, out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        train_ratio=args.train_ratio,
        seed=args.seed,
        step=args.step,
        transform=full_transform 
    )
    logging.info(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # — Model, Optimizer, Losses —
    model = build_temperature_model(args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    logging.info(f"Model size: {get_model_size(model):.2f} MB")

    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    l1_fn       = nn.L1Loss().to(device)
    grad_fn     = multi_scale_grad_loss

    train_losses, val_losses = [], []
    train_errs,   val_errs   = [], []

    best_val_loss = float('inf')

    # — Training Loop —
    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss, run_err = 0.0, 0.0

        for step_idx, batch in enumerate(train_loader, 1):
            p = batch['power'].to(device)
            c = batch['coords'].to(device)
            t = batch['temperature'].to(device)
            
            if step_idx == 1:
                logging.info(f"Batch shapes: power={p.shape}, coords={c.shape}, target={t.shape}")

            optimizer.zero_grad()
            pred = model(p, c)
            mask    = ~torch.isnan(t)
            l1      = l1_fn(pred[mask], t[mask])
            gradl   = grad_fn(pred, t)
            loss    = 0.5*l1 + 0.25*gradl
            loss.backward()
            optimizer.step()

            err = eval_metrics(pred.detach().cpu(), t.detach().cpu())[0]
            run_loss += loss.item()
            run_err  += err

            if step_idx % args.log_step == 0:
                logging.info(f"[E{epoch} S{step_idx}/{len(train_loader)}] Loss={loss:.4f}, Err={err:.4f}")

        avg_tr_loss = run_loss / len(train_loader)
        avg_tr_err  = run_err  / len(train_loader)
        train_losses.append(avg_tr_loss)
        train_errs.append(avg_tr_err)
        writer.add_scalar('train/loss',  avg_tr_loss, epoch)
        writer.add_scalar('train/error', avg_tr_err,   epoch)
        logging.info(f"Epoch {epoch} TRAIN → AvgLoss={avg_tr_loss:.4f}, AvgErr={avg_tr_err:.4f}")

        # — Validation —
        model.eval()
        v_loss, v_err = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                p = batch['power'].to(device)
                c = batch['coords'].to(device)
                t = batch['temperature'].to(device)

                pred = model(p, c)
                mask = ~torch.isnan(t)
                l1   = l1_fn(pred[mask], t[mask])
                gradl= grad_fn(pred, t)
                loss = 0.5*l1 + 0.25*gradl

                err = eval_metrics(pred.cpu(), t.cpu())[0]
                v_loss += loss.item()
                v_err  += err

        avg_val_loss = v_loss / len(val_loader)
        avg_val_err  = v_err  / len(val_loader)
        val_losses.append(avg_val_loss)
        val_errs.append(avg_val_err)
        writer.add_scalar('val/loss',  avg_val_loss, epoch)
        writer.add_scalar('val/error', avg_val_err,   epoch)
        logging.info(f"Epoch {epoch} VAL   → AvgLoss={avg_val_loss:.4f}, AvgErr={avg_val_err:.4f}")

        # — Save intermediate image —
        sample = next(iter(val_loader))
        p0, c0, t0 = sample['power'][0], sample['coords'][0], sample['temperature'][0]
        with torch.no_grad():
            tp = model(p0.unsqueeze(0).to(device), c0.unsqueeze(0).to(device))[0,0].cpu()
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(p0[0].cpu(), cmap='magma');    axes[0].set_title('Input Power')
        axes[1].imshow(t0[0].cpu(), cmap='hot');      axes[1].set_title('True Temp')
        axes[2].imshow(tp,       cmap='hot');         axes[2].set_title('Predicted Temp')
        for ax in axes: ax.axis('off')
        fig.savefig(os.path.join(INTERMEDIATE_DIR, f"epoch_{epoch:03d}.png"))
        plt.close(fig)

        # — Save checkpoint each epoch —
        ckpt = {
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth"))

        # — Save best model —
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, os.path.join(CKPT_DIR, "best.pth"))
            logging.info(f"New best model @ epoch {epoch} (val loss={avg_val_loss:.4f})")

    # — Final plots —
    ep_list = list(range(1, args.epochs+1))
    plt.figure(); plt.plot(ep_list, train_losses, label='Train Loss'); plt.plot(ep_list, val_losses, label='Val Loss')
    plt.savefig(os.path.join(BASE_DIR, 'loss_curve.png')); plt.close()
    plt.figure(); plt.plot(ep_list, train_errs, label='Train Err'); plt.plot(ep_list, val_errs, label='Val Err')
    plt.savefig(os.path.join(BASE_DIR, 'error_curve.png')); plt.close()

    logging.info("Training complete! All artifacts in: " + BASE_DIR)
    print("Done. Checkpoints, logs, tensorboard, images all under", BASE_DIR)

if __name__=='__main__':
    parser = argparse.ArgumentParser('TempTransformer', parents=[get_args_parser()])
    args   = parser.parse_args()
    main(args)
