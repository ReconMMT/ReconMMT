"""ReconMMT evaluation script.

Copyright (c) Gautam Ankoji.
Licensed under the MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.algo import (
    ReconstructionLoss,
    TrainConfig,
    build_dataloader,
    create_model,
    ensure_dir,
    evaluate_model,
    infer_device,
    load_checkpoint,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ReconMMT checkpoints.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        split_test=args.split,
    )
    config.model.image_size = args.image_size
    config.model.base_channels = args.base_channels
    config.model.bottleneck_heads = args.heads
    config.model.bottleneck_layers = args.layers

    device = infer_device(config.device)
    model = create_model(config).to(device)
    load_checkpoint(args.checkpoint, model)
    criterion = ReconstructionLoss(config.model).to(device)

    dataloader = build_dataloader(
        root=config.data_root,
        split=config.split_test,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        augment=False,
    )
    save_dir = ensure_dir(Path(args.save_dir)) if args.save_dir else None
    metrics = evaluate_model(model, dataloader, criterion, device, save_dir=save_dir)

    print(
        f"Evaluation on split={config.split_test} | "
        f"loss={metrics['loss']:.4f} | "
        f"psnr={metrics['psnr']:.2f} | "
        f"ssim={metrics['ssim']:.4f} | "
        f"mae={metrics['mae']:.4f}"
    )
    if save_dir is not None:
        save_json(save_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
