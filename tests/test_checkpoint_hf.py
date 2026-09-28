"""
Unit tests for Hugging Face Hub checkpoint resolution, caching, and from_pretrained loading.
"""

import os
import tempfile
import pytest
import torch

from sam3.model_builder import (
    get_default_hf_cache_dir,
    is_hf_repo_identifier,
    parse_hf_identifier,
    resolve_checkpoint_path,
    download_ckpt_from_hf,
    from_pretrained,
    build_efficientsam3_image_model,
)
from sam3.model.sam3_image import Sam3Image


def test_is_hf_repo_identifier():
    assert is_hf_repo_identifier("facebook/sam3") is True
    assert is_hf_repo_identifier("Simon7108528/EfficientSAM3") is True
    assert is_hf_repo_identifier("hf://facebook/sam3/sam3.pt") is True
    assert is_hf_repo_identifier("https://huggingface.co/Simon7108528/EfficientSAM3") is True
    assert is_hf_repo_identifier("Simon7108528/EfficientSAM3:efficientsam3_tinyvit.pt") is True

    # Local non-repo paths
    assert is_hf_repo_identifier(None) is False
    assert is_hf_repo_identifier("") is False
    assert is_hf_repo_identifier("/workspace/checkpoints/model.pt") is False
    assert is_hf_repo_identifier("./checkpoints/model.pt") is False
    assert is_hf_repo_identifier("model.pt") is False


def test_parse_hf_identifier():
    repo, filename = parse_hf_identifier("Simon7108528/EfficientSAM3")
    assert repo == "Simon7108528/EfficientSAM3"
    assert filename is None

    repo, filename = parse_hf_identifier("Simon7108528/EfficientSAM3:efficientsam3_ft/efficientsam3_tinyvit.pt")
    assert repo == "Simon7108528/EfficientSAM3"
    assert filename == "efficientsam3_ft/efficientsam3_tinyvit.pt"

    repo, filename = parse_hf_identifier("hf://Simon7108528/EfficientSAM3/efficientsam3_ft/efficientsam3_tinyvit.pt")
    assert repo == "Simon7108528/EfficientSAM3"
    assert filename == "efficientsam3_ft/efficientsam3_tinyvit.pt"


def test_resolve_checkpoint_path_local():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"dummy_weights")
        temp_path = f.name

    try:
        resolved = resolve_checkpoint_path(temp_path)
        assert resolved == os.path.abspath(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_resolve_checkpoint_path_hf_mock(monkeypatch):
    recorded_calls = []

    def mock_hf_download(repo_id, filename, **kwargs):
        recorded_calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return f"/mock_cache/{repo_id}/{filename}"

    monkeypatch.setattr("sam3.model_builder.hf_hub_download", mock_hf_download)

    # 1. HF Repo ID with explicit backbone
    res = resolve_checkpoint_path("Simon7108528/EfficientSAM3", backbone_type="tinyvit")
    assert res == "/mock_cache/Simon7108528/EfficientSAM3/efficientsam3_ft/efficientsam3_tinyvit.pt"
    assert recorded_calls[-1]["repo_id"] == "Simon7108528/EfficientSAM3"
    assert recorded_calls[-1]["filename"] == "efficientsam3_ft/efficientsam3_tinyvit.pt"

    # 2. Known checkpoint filename fallback when missing locally
    res2 = resolve_checkpoint_path("checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt")
    assert res2 == "/mock_cache/Simon7108528/EfficientSAM3/efficientsam3_ft/efficientsam3_tinyvit.pt"

    # 3. None or empty
    assert resolve_checkpoint_path(None) is None
    assert resolve_checkpoint_path("none") is None


def test_download_ckpt_from_hf_mock(monkeypatch):
    recorded_calls = []

    def mock_hf_download(repo_id, filename, **kwargs):
        recorded_calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return f"/mock_cache/{repo_id}/{filename}"

    monkeypatch.setattr("sam3.model_builder.hf_hub_download", mock_hf_download)

    p = download_ckpt_from_hf(backbone_type="efficientvit")
    assert p == "/mock_cache/Simon7108528/EfficientSAM3/efficientsam3_ft/efficientsam3_efficientvit.pt"

    p_sam3 = download_ckpt_from_hf()
    assert p_sam3 == "/mock_cache/facebook/sam3/sam3.pt"


def test_from_pretrained_dispatch(monkeypatch):
    built = {}

    def mock_build_eff(checkpoint_path=None, backbone_type="tinyvit", **kwargs):
        built["eff"] = {"checkpoint_path": checkpoint_path, "backbone_type": backbone_type, **kwargs}
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr("sam3.model_builder.build_efficientsam3_image_model", mock_build_eff)

    model = from_pretrained("Simon7108528/EfficientSAM3", backbone_type="tinyvit", eval_mode=True)
    assert isinstance(model, torch.nn.Linear)
    assert built["eff"]["backbone_type"] == "tinyvit"
    assert built["eff"]["checkpoint_path"] == "Simon7108528/EfficientSAM3"


def test_sam3_image_from_pretrained_classmethod():
    assert hasattr(Sam3Image, "from_pretrained")
    assert callable(Sam3Image.from_pretrained)
