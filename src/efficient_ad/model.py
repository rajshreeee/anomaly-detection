import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


def get_pdn_small(out_channels: int = 384, padding: bool = False) -> nn.Sequential:
    p = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(3,   128, kernel_size=4, padding=3*p), nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=p),
        nn.Conv2d(128, 256, kernel_size=4, padding=3*p), nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=p),
        nn.Conv2d(256, 256, kernel_size=3, padding=p),   nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=4),
    )


def get_autoencoder(out_channels: int = 384) -> nn.Sequential:
    return nn.Sequential(
        # encoder
        nn.Conv2d(3,  32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
        # decoder
        nn.Upsample(size=3,   mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=8,   mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=15,  mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=32,  mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=63,  mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Upsample(size=56,  mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, out_channels, 3, 1, 1),
    )


def load_teacher(weights_path: str, out_channels: int, device: str) -> nn.Module:
    teacher = get_pdn_small(out_channels)
    state   = torch.load(weights_path, map_location='cpu')
    teacher.load_state_dict(state)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Teacher loaded from {weights_path}")
    return teacher


@torch.no_grad()
def predict_map(
    image:       torch.Tensor,
    teacher:     nn.Module,
    student:     nn.Module,
    autoencoder: nn.Module,
    t_mean:      torch.Tensor,
    t_std:       torch.Tensor,
    out_channels: int,
    q_st_start:  torch.Tensor,
    q_st_end:    torch.Tensor,
    q_ae_start:  torch.Tensor,
    q_ae_end:    torch.Tensor,
) -> torch.Tensor:
    t_out  = (teacher(image) - t_mean) / t_std
    s_out  = student(image)
    ae_out = autoencoder(image)

    map_st = torch.mean((t_out - s_out[:, :out_channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((ae_out - s_out[:, out_channels:]) ** 2, dim=1, keepdim=True)

    map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    return 0.5 * map_st + 0.5 * map_ae
