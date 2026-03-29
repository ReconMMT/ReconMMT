"""ReconMMT research entrypoint.

Copyright (c) Gautam Ankoji.
Licensed under the MIT License.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from models.algo import (
    ReconMMT,
    ReconMMTConfig,
    TrainConfig,
    export_onnx,
    infer_device,
    load_checkpoint,
    save_image_tensor,
    smoke_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ReconMMT research entrypoint for training, testing, inference, and export."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training via train.py")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    test_parser = subparsers.add_parser("test", help="Run evaluation via test.py")
    test_parser.add_argument("args", nargs=argparse.REMAINDER)

    infer_parser = subparsers.add_parser("infer", help="Run single-sample inference.")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--rgb", required=True)
    infer_parser.add_argument("--thermal", required=True)
    infer_parser.add_argument("--depth", required=True)
    infer_parser.add_argument("--output", default="results/reconmmt/inference.png")
    infer_parser.add_argument("--image-size", type=int, default=256)
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--base-channels", type=int, default=32)
    infer_parser.add_argument("--heads", type=int, default=4)
    infer_parser.add_argument("--layers", type=int, default=2)

    smoke_parser = subparsers.add_parser("smoke-test", help="Verify the model graph and losses.")
    smoke_parser.add_argument("--image-size", type=int, default=64)

    export_parser = subparsers.add_parser("export-onnx", help="Export a checkpoint to ONNX.")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", default="results/reconmmt/reconmmt.onnx")
    export_parser.add_argument("--image-size", type=int, default=256)
    export_parser.add_argument("--base-channels", type=int, default=32)
    export_parser.add_argument("--heads", type=int, default=4)
    export_parser.add_argument("--layers", type=int, default=2)

    return parser


def _load_input(path: str, mode: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert(mode)
    image = TF.resize(image, [image_size, image_size])
    tensor = TF.to_tensor(image) * 2.0 - 1.0
    return tensor.unsqueeze(0)


def _build_model(image_size: int, base_channels: int, heads: int, layers: int) -> ReconMMT:
    config = ReconMMTConfig(
        image_size=image_size,
        base_channels=base_channels,
        bottleneck_heads=heads,
        bottleneck_layers=layers,
    )
    return ReconMMT(config)


def _delegate(script_name: str, extra_args: list[str]) -> None:
    command = [sys.executable, script_name, *extra_args]
    raise SystemExit(subprocess.call(command))


def run_inference(args: argparse.Namespace) -> None:
    device = infer_device(args.device)
    model = _build_model(args.image_size, args.base_channels, args.heads, args.layers).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    rgb = _load_input(args.rgb, "RGB", args.image_size).to(device)
    thermal = _load_input(args.thermal, "L", args.image_size).to(device)
    depth = _load_input(args.depth, "L", args.image_size).to(device)

    with torch.no_grad():
        prediction = model(rgb, thermal, depth)[0]

    save_image_tensor(prediction, Path(args.output))
    print(f"Saved inference result to {args.output}")


def run_smoke_test(args: argparse.Namespace) -> None:
    metrics = smoke_test()
    print(
        f"Smoke test passed | params={int(metrics['parameters'])} | "
        f"output={int(metrics['output_height'])}x{int(metrics['output_width'])} | "
        f"loss={metrics['total']:.4f}"
    )


def run_export(args: argparse.Namespace) -> None:
    model = _build_model(args.image_size, args.base_channels, args.heads, args.layers)
    load_checkpoint(args.checkpoint, model)
    export_onnx(model, args.output, image_size=args.image_size)
    print(f"Exported ONNX model to {args.output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _delegate("train.py", args.args)
    if args.command == "test":
        _delegate("test.py", args.args)
    if args.command == "infer":
        run_inference(args)
        return
    if args.command == "smoke-test":
        run_smoke_test(args)
        return
    if args.command == "export-onnx":
        run_export(args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
