"""
Unit tests for text encoder prompt caching in SAM3VLBackbone and TextStudentEncoder.
"""

import pytest
import torch

from sam3.model_builder import build_efficientsam3_image_model


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_text_encoder_caching(device):
    """Verify that repeat calls to forward_text hit the cache and return identical tensors."""
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

    backbone = model.backbone
    assert hasattr(backbone, "_text_cache")
    backbone.clear_text_cache()
    assert len(backbone._text_cache) == 0

    # 1. First call computes and caches
    out1 = backbone.forward_text(["dog"], device=device)
    assert len(backbone._text_cache) == 1
    assert "language_features" in out1
    assert "language_mask" in out1

    # 2. Second call with same prompt returns cached tensors
    out2 = backbone.forward_text(["dog"], device=device)
    assert len(backbone._text_cache) == 1
    assert torch.equal(out1["language_features"], out2["language_features"])
    assert torch.equal(out1["language_mask"], out2["language_mask"])

    # 3. Different prompt adds a second entry to cache
    out3 = backbone.forward_text(["cat"], device=device)
    assert len(backbone._text_cache) == 2
    assert not torch.equal(out1["language_features"], out3["language_features"])

    # 4. clear_text_cache empties the cache
    backbone.clear_text_cache()
    assert len(backbone._text_cache) == 0
