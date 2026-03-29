"""Core ReconMMT model, dataset, augmentation, and evaluation utilities.

Copyright (c) Gautam Ankoji.
Licensed under the MIT License.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ReconMMTConfig:
    model_name: str = "ReconMMT"
    image_size: int = 256
    rgb_channels: int = 3
    thermal_channels: int = 1
    depth_channels: int = 1
    base_channels: int = 32
    bottleneck_heads: int = 4
    bottleneck_layers: int = 2
    dropout: float = 0.1
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.4
    lambda_edge: float = 0.2


@dataclass
class TrainConfig:
    data_root: str = "data"
    output_dir: str = "results/reconmmt"
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    val_interval: int = 1
    grad_clip: float = 1.0
    seed: int = 42
    mixed_precision: bool = True
    save_images: bool = False
    device: Optional[str] = None
    pin_memory: bool = False
    split_train: str = "train"
    split_val: str = "val"
    split_test: str = "test"
    model: ReconMMTConfig = field(default_factory=ReconMMTConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) * 0.5
    tensor = (tensor * 255.0).byte()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return TF.to_pil_image(tensor)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    ensure_dir(path.parent)
    tensor_to_image(tensor).save(path)


def _normalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0


def _resize_image(image: Image.Image, size: int, interpolation: InterpolationMode) -> Image.Image:
    return TF.resize(image, [size, size], interpolation=interpolation)


def _find_directory(root: Path, aliases: Sequence[str]) -> Optional[Path]:
    for alias in aliases:
        candidate = root / alias
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _scan_images(directory: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


class SynchronizedAugmentation:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(
        self,
        rgb: Image.Image,
        thermal: Image.Image,
        depth: Image.Image,
        target: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        images = [rgb, thermal, depth, target]

        if random.random() < 0.5:
            images = [TF.hflip(image) for image in images]
        if random.random() < 0.2:
            images = [TF.vflip(image) for image in images]

        angle = random.choice([0, 0, 0, 90, 180, 270])
        if angle:
            images = [TF.rotate(image, angle) for image in images]

        crop_scale = random.uniform(0.75, 1.0)
        crop_ratio = random.uniform(0.9, 1.1)
        i, j, h, w = RandomResizedCrop.get_params(
            images[0], scale=(crop_scale, 1.0), ratio=(crop_ratio, crop_ratio)
        )
        images = [
            TF.resized_crop(
                image,
                i,
                j,
                h,
                w,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR
                if image.mode == "RGB"
                else InterpolationMode.BILINEAR,
            )
            for image in images
        ]

        rgb = images[0]
        if random.random() < 0.5:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_contrast(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.85, 1.15))
        if random.random() < 0.25:
            rgb = TF.gaussian_blur(rgb, kernel_size=3, sigma=[0.1, 1.4])
        if random.random() < 0.25:
            fog = Image.new("RGB", rgb.size, color=(235, 235, 235))
            rgb = Image.blend(rgb, fog, alpha=random.uniform(0.08, 0.18))

        return rgb, images[1], images[2], images[3]


class ReconMMTDataset(Dataset):
    directory_aliases = {
        "rgb": ("rgb", "image", "images", "input", "inputs", "degraded"),
        "thermal": ("thermal", "ir", "infrared"),
        "depth": ("depth", "disp", "disparity"),
        "target": ("target", "targets", "gt", "clean", "label", "labels", "rgb_gt"),
    }

    def __init__(self, root: str | Path, split: str, image_size: int, augment: bool = False) -> None:
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        self.image_size = image_size
        self.augment = augment
        self.augmenter = SynchronizedAugmentation(image_size) if augment else None

        if not self.split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory: {self.split_root}. "
                "Expected data/<split>/{rgb,thermal,depth,target}."
            )

        modality_dirs = {
            key: _find_directory(self.split_root, aliases)
            for key, aliases in self.directory_aliases.items()
        }
        missing = [name for name, directory in modality_dirs.items() if directory is None]
        if missing:
            raise FileNotFoundError(
                f"Missing modality folders under {self.split_root}: {', '.join(missing)}"
            )

        modality_files = {name: _scan_images(directory) for name, directory in modality_dirs.items()}
        common_ids = sorted(set.intersection(*(set(files.keys()) for files in modality_files.values())))
        if not common_ids:
            raise RuntimeError(
                f"No paired samples found in {self.split_root}. "
                "Make sure file stems match across modalities."
            )

        self.samples = [
            {name: modality_files[name][sample_id] for name in modality_files}
            for sample_id in common_ids
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, path: Path, mode: str) -> torch.Tensor:
        image = Image.open(path).convert(mode)
        if mode == "RGB":
            image = _resize_image(image, self.image_size, InterpolationMode.BILINEAR)
        else:
            image = _resize_image(image, self.image_size, InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(image)
        return _normalize_rgb(tensor)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        rgb_image = Image.open(sample["rgb"]).convert("RGB")
        thermal_image = Image.open(sample["thermal"]).convert("L")
        depth_image = Image.open(sample["depth"]).convert("L")
        target_image = Image.open(sample["target"]).convert("RGB")

        if self.augmenter is not None:
            rgb_image, thermal_image, depth_image, target_image = self.augmenter(
                rgb_image, thermal_image, depth_image, target_image
            )
        else:
            rgb_image = _resize_image(rgb_image, self.image_size, InterpolationMode.BILINEAR)
            thermal_image = _resize_image(thermal_image, self.image_size, InterpolationMode.BILINEAR)
            depth_image = _resize_image(depth_image, self.image_size, InterpolationMode.BILINEAR)
            target_image = _resize_image(target_image, self.image_size, InterpolationMode.BILINEAR)

        rgb = _normalize_rgb(TF.to_tensor(rgb_image))
        thermal = _normalize_rgb(TF.to_tensor(thermal_image))
        depth = _normalize_rgb(TF.to_tensor(depth_image))
        target = _normalize_rgb(TF.to_tensor(target_image))

        return {
            "rgb": rgb,
            "thermal": thermal,
            "depth": depth,
            "target": target,
            "sample_id": sample["rgb"].stem,
        }


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0]),
            ResidualConvBlock(widths[0]),
        )
        self.down1 = nn.Sequential(ConvNormAct(widths[0], widths[1], stride=2), ResidualConvBlock(widths[1]))
        self.down2 = nn.Sequential(ConvNormAct(widths[1], widths[2], stride=2), ResidualConvBlock(widths[2]))
        self.down3 = nn.Sequential(ConvNormAct(widths[2], widths[3], stride=2), ResidualConvBlock(widths[3]))
        self.widths = widths

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        return [f1, f2, f3, f4]


class GatedFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 3, 1),
        )
        self.refine = ResidualConvBlock(channels)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        weights = self.gate(torch.cat([rgb, thermal, depth], dim=1))
        weights = torch.softmax(weights, dim=1)
        fused = (
            weights[:, 0:1] * rgb
            + weights[:, 1:2] * thermal
            + weights[:, 2:3] * depth
        )
        return self.refine(fused)


class TransformerBottleneckFusion(nn.Module):
    def __init__(self, channels: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rgb_to_thermal = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.rgb_to_depth = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_proj = nn.Linear(channels * 3, channels)
        self.norm = nn.LayerNorm(channels)

    def _flatten(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        return feature_map.flatten(2).transpose(1, 2), height, width

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_tokens, height, width = self._flatten(rgb)
        thermal_tokens, _, _ = self._flatten(thermal)
        depth_tokens, _, _ = self._flatten(depth)

        rgb_tokens = self.norm(rgb_tokens)
        thermal_tokens = self.norm(thermal_tokens)
        depth_tokens = self.norm(depth_tokens)

        rgb_from_thermal, _ = self.rgb_to_thermal(rgb_tokens, thermal_tokens, thermal_tokens)
        rgb_from_depth, _ = self.rgb_to_depth(rgb_tokens, depth_tokens, depth_tokens)
        fused = torch.cat([rgb_tokens, rgb_from_thermal, rgb_from_depth], dim=-1)
        fused = self.output_proj(fused)
        fused = self.transformer(fused)
        return fused.transpose(1, 2).reshape(rgb.shape[0], rgb.shape[1], height, width)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ReconMMT(nn.Module):
    def __init__(self, config: Optional[ReconMMTConfig] = None) -> None:
        super().__init__()
        self.config = config or ReconMMTConfig()
        base = self.config.base_channels

        self.rgb_encoder = ModalityEncoder(self.config.rgb_channels, base)
        self.thermal_encoder = ModalityEncoder(self.config.thermal_channels, base)
        self.depth_encoder = ModalityEncoder(self.config.depth_channels, base)

        widths = self.rgb_encoder.widths
        self.fusion_blocks = nn.ModuleList(
            [GatedFusion(widths[0]), GatedFusion(widths[1]), GatedFusion(widths[2])]
        )
        self.bottleneck_fusion = TransformerBottleneckFusion(
            widths[3],
            heads=self.config.bottleneck_heads,
            layers=self.config.bottleneck_layers,
            dropout=self.config.dropout,
        )

        self.dec3 = DecoderBlock(widths[3], widths[2], widths[2])
        self.dec2 = DecoderBlock(widths[2], widths[1], widths[1])
        self.dec1 = DecoderBlock(widths[1], widths[0], widths[0])
        self.head = nn.Sequential(
            ConvNormAct(widths[0], widths[0]),
            nn.Conv2d(widths[0], 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_encoder(rgb)
        thermal_feats = self.thermal_encoder(thermal)
        depth_feats = self.depth_encoder(depth)

        fused_feats = [
            fusion(rgb_level, thermal_level, depth_level)
            for fusion, rgb_level, thermal_level, depth_level in zip(
                self.fusion_blocks, rgb_feats[:3], thermal_feats[:3], depth_feats[:3]
            )
        ]
        bottleneck = self.bottleneck_fusion(rgb_feats[3], thermal_feats[3], depth_feats[3])

        x = self.dec3(bottleneck, fused_feats[2])
        x = self.dec2(x, fused_feats[1])
        x = self.dec1(x, fused_feats[0])
        residual = F.interpolate(rgb, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x) * 0.7 + residual * 0.3

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


MultiModalReconstructionModel = ReconMMT


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
        kernel = self._create_kernel(window_size)
        self.register_buffer("kernel", kernel)

    def _create_kernel(self, window_size: int) -> torch.Tensor:
        gauss = torch.tensor(
            [math.exp(-((value - window_size // 2) ** 2) / (2 * 1.5 ** 2)) for value in range(window_size)],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        kernel = torch.outer(gauss, gauss).view(1, 1, window_size, window_size)
        return kernel.repeat(3, 1, 1, 1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = (prediction + 1.0) * 0.5
        target = (target + 1.0) * 0.5
        padding = self.window_size // 2
        kernel = self.kernel.to(prediction.device, prediction.dtype)
        mu_pred = F.conv2d(prediction, kernel, padding=padding, groups=3)
        mu_target = F.conv2d(target, kernel, padding=padding, groups=3)
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred = F.conv2d(prediction * prediction, kernel, padding=padding, groups=3) - mu_pred_sq
        sigma_target = F.conv2d(target * target, kernel, padding=padding, groups=3) - mu_target_sq
        sigma_cross = F.conv2d(prediction * target, kernel, padding=padding, groups=3) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.c1) * (2 * sigma_cross + self.c2)) / (
            (mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred + sigma_target + self.c2)
        )
        return 1.0 - ssim_map.mean()


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(x.device, x.dtype), padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y.to(x.device, x.dtype), padding=1, groups=3)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient(prediction), self._gradient(target))


class ReconstructionLoss(nn.Module):
    def __init__(self, config: ReconMMTConfig) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.config = config

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_value = self.l1(prediction, target)
        ssim_value = self.ssim(prediction, target)
        edge_value = self.edge(prediction, target)
        total = (
            self.config.lambda_l1 * l1_value
            + self.config.lambda_ssim * ssim_value
            + self.config.lambda_edge * edge_value
        )
        return total, {
            "l1": float(l1_value.detach().cpu()),
            "ssim": float(ssim_value.detach().cpu()),
            "edge": float(edge_value.detach().cpu()),
            "total": float(total.detach().cpu()),
        }


@torch.no_grad()
def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.clamp(-1.0, 1.0)
    target = target.clamp(-1.0, 1.0)
    mse = F.mse_loss(prediction, target)
    return float(10.0 * torch.log10(torch.tensor(4.0, device=prediction.device) / mse).cpu())


@torch.no_grad()
def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(1.0 - SSIMLoss()(prediction, target).detach().cpu())


@torch.no_grad()
def compute_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.l1_loss(prediction, target).detach().cpu())


def build_dataloader(
    root: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = ReconMMTDataset(root=root, split=split, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_model(train_config: TrainConfig) -> ReconMMT:
    model_config = train_config.model
    model_config.image_size = train_config.image_size
    return ReconMMT(model_config)


def save_checkpoint(
    path: Path,
    model: ReconMMT,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    train_config: TrainConfig,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "train_config": asdict(train_config),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def export_onnx(model: ReconMMT, output_path: str, image_size: int = 256) -> None:
    model.eval()
    dummy_rgb = torch.randn(1, 3, image_size, image_size)
    dummy_thermal = torch.randn(1, 1, image_size, image_size)
    dummy_depth = torch.randn(1, 1, image_size, image_size)
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_thermal, dummy_depth),
        output_path,
        input_names=["rgb", "thermal", "depth"],
        output_names=["reconstruction"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "thermal": {0: "batch"},
            "depth": {0: "batch"},
            "reconstruction": {0: "batch"},
        },
        opset_version=17,
    )


def run_model_step(
    model: ReconMMT,
    batch: Dict[str, torch.Tensor | str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(device)  # type: ignore[union-attr]
    thermal = batch["thermal"].to(device)  # type: ignore[union-attr]
    depth = batch["depth"].to(device)  # type: ignore[union-attr]
    target = batch["target"].to(device)  # type: ignore[union-attr]
    prediction = model(rgb, thermal, depth)
    return prediction, target


def evaluate_model(
    model: ReconMMT,
    dataloader: DataLoader,
    criterion: Optional[ReconstructionLoss],
    device: torch.device,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    mae_scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            prediction, target = run_model_step(model, batch, device)
            if criterion is not None:
                loss, _ = criterion(prediction, target)
                losses.append(float(loss.detach().cpu()))

            for index in range(prediction.shape[0]):
                pred_sample = prediction[index : index + 1]
                target_sample = target[index : index + 1]
                psnr_scores.append(compute_psnr(pred_sample, target_sample))
                ssim_scores.append(compute_ssim(pred_sample, target_sample))
                mae_scores.append(compute_mae(pred_sample, target_sample))

                if save_dir is not None:
                    sample_id = batch["sample_id"][index]  # type: ignore[index]
                    save_image_tensor(prediction[index], save_dir / f"{sample_id}_pred.png")

    metrics = {
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "psnr": sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0,
        "ssim": sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0,
        "mae": sum(mae_scores) / len(mae_scores) if mae_scores else 0.0,
    }
    return metrics


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def smoke_test() -> Dict[str, float]:
    config = ReconMMTConfig(image_size=64, base_channels=16, bottleneck_heads=2, bottleneck_layers=1)
    model = ReconMMT(config)
    rgb = torch.randn(2, 3, 64, 64)
    thermal = torch.randn(2, 1, 64, 64)
    depth = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 3, 64, 64).clamp(-1.0, 1.0)

    prediction = model(rgb, thermal, depth)
    criterion = ReconstructionLoss(config)
    loss, breakdown = criterion(prediction, target)
    loss.backward()

    return {
        "parameters": model.num_parameters(),
        "output_height": float(prediction.shape[-2]),
        "output_width": float(prediction.shape[-1]),
        **breakdown,
    }
