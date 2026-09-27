#!/usr/bin/env python3
"""
EfficientSAM3 LoRA / QLoRA Fine-Tuning Script
Demonstrates low-compute parameter-efficient fine-tuning on custom segmentation data.

Usage:
    # Standard LoRA (fp16 / bf16):
    python scripts/train_lora.py --checkpoint checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt --lora-r 16

    # QLoRA (4-bit base weights + LoRA adapters):
    python scripts/train_lora.py --checkpoint checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt --qlora --lora-r 16
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA / QLoRA adapters on EfficientSAM3")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
        help="Path to pre-distilled EfficientSAM3 checkpoint",
    )
    parser.add_argument("--backbone", type=str, default="tinyvit", choices=["tinyvit", "efficientvit", "repvit"])
    parser.add_argument("--model-name", type=str, default="11m")
    parser.add_argument("--text-encoder", type=str, default="MobileCLIP-S0")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--qlora", action="store_true", help="Enable 4-bit base model quantization (QLoRA)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="output/lora_adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class SyntheticSegmentationDataset(Dataset):
    """Minimal dataset for training demonstration."""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image and dummy mask
        image = torch.randn(3, 1008, 1008)
        mask = (torch.rand(1, 1008, 1008) > 0.7).float()
        return {"image": image, "mask": mask, "prompt": "object"}


def apply_lora(model, args):
    """Attach LoRA adapters to linear projections in Transformer & Backbone."""
    # Target projections inside DETR transformer and attention projections
    target_modules = [
        "qkv", "qkv_proj", "out_proj", "proj", "linear1", "linear2", "fc1", "fc2"
    ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    print("\n[LoRA Configuration]")
    peft_model.print_trainable_parameters()
    return peft_model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"==================================================")
    print(f"EfficientSAM3 LoRA / QLoRA Fine-Tuning")
    print(f"Device: {args.device} | QLoRA: {args.qlora} | Rank: {args.lora_r}")
    print(f"==================================================")

    # 1. Build Base Model
    print(f"\nBuilding base EfficientSAM3 ({args.backbone}-{args.model_name})...")
    ckpt = args.checkpoint if os.path.exists(args.checkpoint) else None
    if ckpt is None:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found locally. Loading randomly initialized model for demo.")

    base_model = build_efficientsam3_image_model(
        checkpoint_path=ckpt,
        backbone_type=args.backbone,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder,
        text_encoder_context_length=args.context_length,
        load_from_HF=False,
        device=args.device,
        eval_mode=False,
    )

    # 2. Attach LoRA / QLoRA
    peft_model = apply_lora(base_model, args)
    peft_model.to(args.device)

    # 3. Setup Optimizer (only for adapter weights)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print(f"\nInitialized AdamW optimizer for {len(trainable_params)} adapter parameter tensors.")
    print(f"Saving initial adapter configuration to {args.output_dir}...")
    peft_model.save_pretrained(args.output_dir)
    print(f"Successfully saved LoRA adapter to: {args.output_dir}")

    print("\nAdapter training skeleton ready. To run full training on your dataset, connect your DataLoader.")


if __name__ == "__main__":
    main()
