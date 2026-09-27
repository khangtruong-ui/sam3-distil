"""
Tests for LoRA (Low-Rank Adaptation) adapter injection on EfficientSAM3.
Verifies parameter efficiency and adapter attach/detach using Hugging Face PEFT.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from sam3.model_builder import build_efficientsam3_image_model


def test_lora_adapter_injection():
    """Verify that PEFT LoRA attaches properly to EfficientSAM3 linear projections."""
    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        device="cpu",
        eval_mode=False,
    )

    # Configure LoRA targeting transformer attention & feed-forward projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "qkv_proj", "out_proj", "proj", "linear1", "linear2"],
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    # Trainable parameters should be < 3% of the total model
    assert trainable_params > 0, "There should be trainable LoRA parameters"
    assert trainable_pct < 3.0, f"Expected < 3% trainable params, got {trainable_pct:.2f}%"
    assert trainable_params < 3e6, f"Expected < 3M trainable params, got {trainable_params / 1e6:.2f}M"


def test_lora_trainable_gradients():
    """Verify that gradients only compute for LoRA adapter weights, keeping base frozen."""
    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        device="cpu",
        eval_mode=False,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["out_proj", "linear1"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # Check that base parameters are frozen and lora parameters require grad
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA param {name} should require grad"
        else:
            assert param.requires_grad is False, f"Base param {name} should be frozen"
