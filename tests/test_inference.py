"""
Integration tests for EfficientSAM3 image inference and prompt segmentation.
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@pytest.fixture(scope="module")
def sample_image():
    """Load or generate a sample RGB image for testing."""
    test_img_path = "assets/dog_person.jpeg"
    if os.path.exists(test_img_path):
        return Image.open(test_img_path).convert("RGB")
    # Synthetic fallback
    arr = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tinyvit_model(device):
    ckpt_path = "checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}, skipping live inference test.")
    
    model = build_efficientsam3_image_model(
        checkpoint_path=ckpt_path,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    return model


def test_image_encoding(tinyvit_model, sample_image):
    """Verify that image feature extraction succeeds and returns valid state."""
    processor = Sam3Processor(tinyvit_model)
    state = processor.set_image(sample_image)
    assert state is not None
    assert "backbone_out" in state or "img_feats" in state or len(state) > 0


def test_text_prompt_segmentation(tinyvit_model, sample_image):
    """Verify text-prompted segmentation produces valid masks and scores."""
    processor = Sam3Processor(tinyvit_model)
    state = processor.set_image(sample_image)
    result = processor.set_text_prompt("dog", state)

    assert "masks" in result
    assert "scores" in result
    assert "boxes" in result

    masks = result["masks"]
    scores = result["scores"]
    boxes = result["boxes"]

    assert isinstance(masks, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert isinstance(boxes, torch.Tensor)

    # Validate score range [0, 1]
    if len(scores) > 0:
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
        # Verify mask spatial dimensions match image or expected resolution
        assert masks.ndim == 3 or masks.ndim == 4
        # Validate boxes have 4 coordinates
        assert boxes.shape[-1] == 4
