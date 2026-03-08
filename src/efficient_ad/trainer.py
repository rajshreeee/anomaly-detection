# src/efficientad/trainer.py

import logging
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def teacher_normalization(
    teacher: nn.Module,
    loader,
    device: str,
) -> tuple:
    """
    Compute channel-wise mean and std of teacher outputs over the training set.
    Used to standardize teacher features before computing ST loss.
    Must be computed AFTER loading teacher weights and BEFORE training.
    """
    means = []
    for img, _ in tqdm(loader, desc='Teacher mean'):
        out = teacher(img.to(device))
        means.append(torch.mean(out, dim=[0, 2, 3]))
    t_mean = torch.mean(torch.stack(means), dim=0)[None, :, None, None]

    vars_ = []
    for img, _ in tqdm(loader, desc='Teacher std'):
        out = teacher(img.to(device))
        vars_.append(torch.mean((out - t_mean) ** 2, dim=[0, 2, 3]))
    t_std = torch.sqrt(torch.mean(torch.stack(vars_), dim=0))[None, :, None, None]

    logger.info("Teacher normalization complete")
    return t_mean, t_std


@torch.no_grad()
def map_normalization(
    val_loader,
    teacher:     nn.Module,
    student:     nn.Module,
    autoencoder: nn.Module,
    t_mean:      torch.Tensor,
    t_std:       torch.Tensor,
    out_channels: int,
    device:      str,
) -> tuple:
    """
    Compute quantile bounds of ST and AE anomaly maps over the validation set.
    These bounds are used to scale raw anomaly scores to a comparable range
    during inference, so ST and AE maps contribute equally to the final score.
    """
    maps_st, maps_ae = [], []

    for img, _ in tqdm(val_loader, desc='Map normalization'):
        img    = img.to(device)
        t_out  = (teacher(img) - t_mean) / t_std
        s_out  = student(img)
        ae_out = autoencoder(img)

        maps_st.append(torch.mean((t_out - s_out[:, :out_channels]) ** 2,
                                   dim=1, keepdim=True))
        maps_ae.append(torch.mean((ae_out - s_out[:, out_channels:]) ** 2,
                                   dim=1, keepdim=True))

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, 0.9)
    q_st_end   = torch.quantile(maps_st, 0.995)
    q_ae_start = torch.quantile(maps_ae, 0.9)
    q_ae_end   = torch.quantile(maps_ae, 0.995)

    logger.info(f"ST  quantiles: [{q_st_start:.5f}, {q_st_end:.5f}]")
    logger.info(f"AE  quantiles: [{q_ae_start:.5f}, {q_ae_end:.5f}]")
    return q_st_start, q_st_end, q_ae_start, q_ae_end


def train_one_step(
    teacher:      nn.Module,
    student:      nn.Module,
    autoencoder:  nn.Module,
    img_st:       torch.Tensor,
    img_ae:       torch.Tensor,
    t_mean:       torch.Tensor,
    t_std:        torch.Tensor,
    out_channels: int,
    optimizer:    torch.optim.Optimizer,
) -> float:
    """
    Single training step for student + autoencoder.

    Three losses:
      loss_st   — student mimics teacher on img_st (hard example mining via 99.9th percentile)
      loss_ae   — autoencoder reconstructs teacher features on img_ae
      loss_stae — student's AE channels mimic autoencoder output (ties them together)

    Teacher is always frozen — no gradients flow through it.

    Returns total loss as float for logging.
    """
    with torch.no_grad():
        t_out = (teacher(img_st) - t_mean) / t_std

    s_out    = student(img_st)[:, :out_channels]
    dist_st  = (t_out - s_out) ** 2
    loss_st  = torch.mean(dist_st[dist_st >= torch.quantile(dist_st, 0.999)])

    ae_out   = autoencoder(img_ae)
    with torch.no_grad():
        t_out_ae = (teacher(img_ae) - t_mean) / t_std
    s_out_ae  = student(img_ae)[:, out_channels:]
    loss_ae   = torch.mean((t_out_ae - ae_out) ** 2)
    loss_stae = torch.mean((ae_out - s_out_ae) ** 2)

    loss = loss_st + loss_ae + loss_stae

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
