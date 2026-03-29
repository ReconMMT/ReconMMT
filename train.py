"""ReconMMT training script.

Copyright (c) Gautam Ankoji.
Licensed under the MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from models.algo import (
    ReconstructionLoss,
    TrainConfig,
    build_dataloader,
    create_model,
    ensure_dir,
    evaluate_model,
    infer_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReconMMT on paired RGB/thermal/depth data.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/reconmmt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        save_images=args.save_images,
        device=args.device,
        split_train=args.train_split,
        split_val=args.val_split,
    )
    config.model.image_size = args.image_size
    config.model.base_channels = args.base_channels
    config.model.bottleneck_heads = args.heads
    config.model.bottleneck_layers = args.layers
    return config


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total_losses: List[float] = []

    for batch in dataloader:
        rgb = batch["rgb"].to(device)
        thermal = batch["thermal"].to(device)
        depth = batch["depth"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = mixed_precision and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            prediction = model(rgb, thermal, depth)
            loss, _ = criterion(prediction, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_losses.append(float(loss.detach().cpu()))

    return {"loss": sum(total_losses) / len(total_losses)}


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    output_dir = ensure_dir(Path(config.output_dir))
    device = infer_device(config.device)
    model = create_model(config).to(device)
    criterion = ReconstructionLoss(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")

    train_loader = build_dataloader(
        root=config.data_root,
        split=config.split_train,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        augment=True,
        pin_memory=config.pin_memory,
    )
    val_loader = build_dataloader(
        root=config.data_root,
        split=config.split_val,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        augment=False,
        pin_memory=config.pin_memory,
    )

    history: List[Dict[str, float]] = []
    best_psnr = float("-inf")
    best_path = output_dir / "best_reconmmt.pt"
    latest_path = output_dir / "last_reconmmt.pt"

    print(f"Training {config.model.model_name} on {device}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Parameters: {model.num_parameters():,}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=config.grad_clip,
            mixed_precision=config.mixed_precision,
        )
        scheduler.step()

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            save_dir=(output_dir / "val_predictions" / f"epoch_{epoch:03d}") if config.save_images else None,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_mae": val_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} | "
            f"val_psnr={epoch_metrics['val_psnr']:.2f} | "
            f"val_ssim={epoch_metrics['val_ssim']:.4f} | "
            f"val_mae={epoch_metrics['val_mae']:.4f}"
        )

        save_checkpoint(latest_path, model, optimizer, epoch, epoch_metrics, config)
        if epoch_metrics["val_psnr"] > best_psnr:
            best_psnr = epoch_metrics["val_psnr"]
            save_checkpoint(best_path, model, optimizer, epoch, epoch_metrics, config)

    save_json(output_dir / "training_history.json", {"history": history})
    save_json(output_dir / "train_config.json", {"config": config.__dict__ | {"model": config.model.__dict__}})
    print(f"Best validation PSNR: {best_psnr:.2f}")
    print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
