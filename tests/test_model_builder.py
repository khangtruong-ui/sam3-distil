"""
Unit tests for EfficientSAM3 model builders, verifying backbone creation and parameter counts.
"""

import os
import pytest
import torch

from sam3.model_builder import build_efficientsam3_image_model


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_build_tinyvit_architecture(device):
    """Test building EfficientSAM3 with TinyViT-11M backbone."""
    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    assert model is not None
    assert hasattr(model, "backbone")
    assert hasattr(model, "transformer")
    assert hasattr(model, "geometry_encoder")

    # Verify parameter count: ~103M params
    total_params = sum(p.numel() for p in model.parameters())
    vis_params = sum(p.numel() for p in model.backbone.vision_backbone.parameters())
    text_params = sum(p.numel() for p in model.backbone.language_backbone.parameters())

    assert 90e6 < total_params < 120e6, f"Expected total params ~103M, got {total_params / 1e6:.2f}M"
    assert 20e6 < vis_params < 35e6, f"Expected vision params ~28M, got {vis_params / 1e6:.2f}M"
    assert 35e6 < text_params < 50e6, f"Expected text params ~42M, got {text_params / 1e6:.2f}M"


def test_build_efficientvit_architecture(device):
    """Test building EfficientSAM3 with EfficientViT-B1 backbone."""
    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    vis_params = sum(p.numel() for p in model.backbone.vision_backbone.parameters())

    assert 85e6 < total_params < 115e6, f"Expected total params ~97M, got {total_params / 1e6:.2f}M"
    assert 18e6 < vis_params < 30e6, f"Expected vision params ~22M, got {vis_params / 1e6:.2f}M"
