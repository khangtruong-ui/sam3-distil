"""
Unit tests for sam3-distil package structure and basic imports.
"""

import os
import pytest


def test_package_import():
    """Verify that sam3 package can be imported directly."""
    import sam3
    assert hasattr(sam3, "__file__"), "sam3 must have __file__ attribute"
    assert os.path.exists(sam3.__file__), "sam3 module file must exist on disk"


def test_model_builder_import():
    """Verify that build functions are exposed by model_builder."""
    from sam3.model_builder import (
        build_efficientsam3_image_model,
        build_sam3_image_model,
        build_efficientsam3_video_model,
    )
    assert callable(build_efficientsam3_image_model)
    assert callable(build_sam3_image_model)
    assert callable(build_efficientsam3_video_model)


def test_processor_import():
    """Verify Sam3Processor is importable."""
    from sam3.model.sam3_image_processor import Sam3Processor
    assert callable(Sam3Processor)


def test_bpe_vocab_asset_presence():
    """Verify that the BPE vocabulary asset exists and is non-empty."""
    import sam3
    pkg_dir = os.path.dirname(sam3.__file__)
    candidate_paths = [
        os.path.join(pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(pkg_dir, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    ]
    found = any(os.path.isfile(p) and os.path.getsize(p) > 1000 for p in candidate_paths)
    assert found, f"BPE vocabulary asset not found in candidates: {candidate_paths}"
