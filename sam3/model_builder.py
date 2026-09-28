# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.text_encoder_student import TextStudentEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_student_text_encoder(bpe_path: str, backbone_type: str, context_length: int = 32) -> TextStudentEncoder:
    """Create Student text encoder."""
    
    # Default config values
    cfg = {
        "context_length": 77, # MobileCLIP default
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if backbone_type == "MobileCLIP-S0":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 4,
            "n_heads_per_layer": 8,
            "model_name": "mct",
        })
    elif backbone_type in ["MobileCLIP-S1", "MobileCLIP2-S0", "MobileCLIP2-S2"]:
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
        })
    elif backbone_type == "MobileCLIP-B":
        cfg.update({
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
            "causal_masking": True,
        })
    elif backbone_type in ["MobileCLIP2-S3", "MobileCLIP2-S4", "MobileCLIP2-L"]:
        cfg.update({
            "dim": 768,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 12,
            "model_name": "base", 
        })
    
    # Update context length in config to match requested length
    cfg["context_length"] = context_length

    return TextStudentEncoder(
        cfg=cfg,
        context_length=context_length, # Use provided context length (default 32)
        output_dim=256, # SAM3 d_model
        bpe_path=bpe_path
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


logger = logging.getLogger("sam3.model_builder")

DEFAULT_SAM3_REPO = "facebook/sam3"
DEFAULT_SAM3_CKPT = "sam3.pt"
DEFAULT_SAM3_CFG = "config.json"

DEFAULT_EFFICIENTSAM3_REPO = "Simon7108528/EfficientSAM3"

BACKBONE_TO_EFFICIENTSAM3_CKPT = {
    "tinyvit": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tiny_vit": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tv": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tvm": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "efficientvit": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "efficient_vit": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "ev": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "evm": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "repvit": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rep_vit": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rv": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rvm": "efficientsam3_ft/efficientsam3_repvit.pt",
}

KNOWN_CHECKPOINT_FILENAMES = {
    "efficientsam3_tinyvit.pt": (DEFAULT_EFFICIENTSAM3_REPO, "efficientsam3_ft/efficientsam3_tinyvit.pt"),
    "efficientsam3_efficientvit.pt": (DEFAULT_EFFICIENTSAM3_REPO, "efficientsam3_ft/efficientsam3_efficientvit.pt"),
    "efficientsam3_repvit.pt": (DEFAULT_EFFICIENTSAM3_REPO, "efficientsam3_ft/efficientsam3_repvit.pt"),
    "sam3.pt": (DEFAULT_SAM3_REPO, DEFAULT_SAM3_CKPT),
}


def get_default_hf_cache_dir() -> Optional[str]:
    """Return standard Hugging Face cache directory ($HF_HOME/hub or $HF_HUB_CACHE)."""
    if "HF_HUB_CACHE" in os.environ:
        return os.environ["HF_HUB_CACHE"]
    if "HF_HOME" in os.environ:
        return os.path.join(os.environ["HF_HOME"], "hub")
    return None


def is_hf_repo_identifier(identifier: str) -> bool:
    """Determine if a given string represents a Hugging Face repository identifier or URI."""
    if not identifier or not isinstance(identifier, str):
        return False
    raw = identifier.strip()
    if raw.startswith("hf://") or raw.startswith("https://huggingface.co/"):
        return True
    if os.path.exists(raw):
        return False
    if raw.startswith(("/", "./", "../", "checkpoints/", "outputs/", "output/", "assets/", "tests/")) or "\\" in raw:
        return False
    # If it ends with model weight extension and has no URI scheme or ':' syntax, treat as file path
    ext = os.path.splitext(raw)[1].lower()
    if ext in [".pt", ".pth", ".bin", ".safetensors", ".ckpt"] and ":" not in raw:
        return False

    repo_part = raw.split(":", 1)[0]
    parts = repo_part.split("/")
    if len(parts) == 2:
        owner, repo = parts[0], parts[1]
        if re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) and re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
            return True
    return False


def parse_hf_identifier(identifier: str) -> Tuple[str, Optional[str]]:
    """Parse HF URI or identifier into (repo_id, optional_filename)."""
    cleaned = identifier.strip()
    if cleaned.startswith("hf://"):
        cleaned = cleaned[5:]
    elif cleaned.startswith("https://huggingface.co/"):
        cleaned = cleaned[23:]
        cleaned = re.sub(r"^(models/)?", "", cleaned)
        cleaned = re.sub(r"/(resolve|blob)/[^/]+/", "/", cleaned)

    if ":" in cleaned:
        repo_id, filename = cleaned.split(":", 1)
        return repo_id.strip().rstrip("/"), filename.strip().lstrip("/")
    parts = cleaned.rstrip("/").split("/")
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}", None
    elif len(parts) > 2:
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return cleaned, None


def download_ckpt_from_hf(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    backbone_type: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """Download checkpoint from Hugging Face Hub into the standard HF cache directory."""
    effective_cache = cache_dir or get_default_hf_cache_dir()

    if repo_id is None:
        if backbone_type and backbone_type.lower() in BACKBONE_TO_EFFICIENTSAM3_CKPT:
            repo_id = DEFAULT_EFFICIENTSAM3_REPO
            filename = filename or BACKBONE_TO_EFFICIENTSAM3_CKPT[backbone_type.lower()]
        else:
            repo_id = DEFAULT_SAM3_REPO
            filename = filename or DEFAULT_SAM3_CKPT

    if filename is None:
        if repo_id == DEFAULT_SAM3_REPO:
            filename = DEFAULT_SAM3_CKPT
        elif backbone_type and backbone_type.lower() in BACKBONE_TO_EFFICIENTSAM3_CKPT:
            filename = BACKBONE_TO_EFFICIENTSAM3_CKPT[backbone_type.lower()]
        else:
            filename = "efficientsam3_ft/efficientsam3_tinyvit.pt"

    if repo_id == DEFAULT_SAM3_REPO:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=DEFAULT_SAM3_CFG,
                cache_dir=effective_cache,
                token=token,
                revision=revision,
            )
        except Exception:
            pass

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=effective_cache,
        token=token,
        revision=revision,
        force_download=force_download,
    )


def resolve_checkpoint_path(
    checkpoint_path: Optional[str] = None,
    backbone_type: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Optional[str]:
    """Resolve a checkpoint path or Hugging Face Hub reference to a local file path.
    
    Supports:
      - Existing local file paths (absolute or relative)
      - Existing local directories (finds newest/standard checkpoint file)
      - Hugging Face Hub repo IDs (e.g. 'Simon7108528/EfficientSAM3', 'facebook/sam3')
      - Hugging Face Hub file URIs (e.g. 'hf://Simon7108528/EfficientSAM3/efficientsam3_ft/...')
      - Automatic fallback download for known checkpoint filenames when local file does not exist.
      - Cached in standard HF cache folder ($HF_HOME/hub or ~/.cache/huggingface/hub).
    """
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in ("none", "", "null"):
        return None

    raw_path = str(checkpoint_path).strip()

    # 1. Existing local file
    if os.path.isfile(raw_path):
        return os.path.abspath(raw_path)

    # 2. Existing local directory: search for common checkpoint files
    if os.path.isdir(raw_path):
        for cand in ["checkpoint.pt", "model.pt", "pytorch_model.bin", "sam3.pt",
                     "efficientsam3_tinyvit.pt", "efficientsam3_efficientvit.pt", "efficientsam3_repvit.pt"]:
            cand_p = os.path.join(raw_path, cand)
            if os.path.isfile(cand_p):
                return os.path.abspath(cand_p)

    # 3. Check relative to workspace or repository roots
    for base in ["/workspace", "/workspace/sam3-distil", os.getcwd()]:
        cand = os.path.join(base, raw_path.lstrip("/"))
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    effective_cache_dir = cache_dir or get_default_hf_cache_dir()

    # 4. Hugging Face URI or Repo ID
    if is_hf_repo_identifier(raw_path):
        repo_id, target_file = parse_hf_identifier(raw_path)
        if not target_file:
            if backbone_type and backbone_type.lower() in BACKBONE_TO_EFFICIENTSAM3_CKPT:
                target_file = BACKBONE_TO_EFFICIENTSAM3_CKPT[backbone_type.lower()]
            elif "sam3" in repo_id.lower() and "efficient" not in repo_id.lower():
                target_file = DEFAULT_SAM3_CKPT
            else:
                target_file = "efficientsam3_ft/efficientsam3_tinyvit.pt"

        logger.info(
            f"Downloading checkpoint from Hugging Face Hub: {repo_id}/{target_file} "
            f"(cache_dir={effective_cache_dir or 'default'})..."
        )
        return hf_hub_download(
            repo_id=repo_id,
            filename=target_file,
            cache_dir=effective_cache_dir,
            token=token,
            revision=revision,
            force_download=force_download,
        )

    # 5. Check if filename matches a known checkpoint for automated HF Hub fetch
    basename = os.path.basename(raw_path)
    if basename in KNOWN_CHECKPOINT_FILENAMES:
        repo_id, target_file = KNOWN_CHECKPOINT_FILENAMES[basename]
        logger.info(
            f"Local checkpoint '{raw_path}' not found on disk. "
            f"Fetching '{target_file}' from Hugging Face Hub '{repo_id}' into cache..."
        )
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                cache_dir=effective_cache_dir,
                token=token,
                revision=revision,
                force_download=force_download,
            )
        except Exception as exc:
            logger.warning(f"Could not download checkpoint from Hugging Face Hub: {exc}")

    # Fallback to raw_path
    logger.warning(f"Checkpoint path '{checkpoint_path}' not found locally and not a recognized HF identifier.")
    return raw_path


def _load_checkpoint(model, checkpoint_path, backbone_type=None, cache_dir=None, token=None, force_download=False):
    """Load model checkpoint from file or HF Hub."""
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path,
        backbone_type=backbone_type,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        # Check if torch version supports weights_only
        try:
           ckpt = torch.load(f, map_location="cpu", weights_only=True)
        except TypeError:
           ckpt = torch.load(f, map_location="cpu")
           
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
        
    # Standardize keys by removing prefixes and handling wrappers
    cleaned_ckpt = {}
    for k, v in ckpt.items():
        new_k = k
        if new_k.startswith("detector."):
            new_k = new_k.replace("detector.", "")
            
        # Handle 'student_trunk' wrapper which might be present in student usage
        # e.g., ...backbone.model.student_trunk.head... -> ...backbone.model.head...
        if "student_trunk." in new_k:
            new_k = new_k.replace("student_trunk.", "")

        cleaned_ckpt[new_k] = v
        
    sam3_image_ckpt = {
        k: v for k, v in cleaned_ckpt.items() if k in model.state_dict()
    }
    
    # Handle tracker/instance predictor if enabled
    if getattr(model, "inst_interactive_predictor", None) is not None:
        tracker_prefix = "inst_interactive_predictor.model."
        tracker_keys = {
            k.replace("tracker.", tracker_prefix): v 
            for k, v in ckpt.items() 
            if "tracker" in k
        }
        sam3_image_ckpt.update(tracker_keys)

    missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing keys: {len(missing_keys)} and unexpected keys: {len(unexpected_keys)}.\n"
            f"Sample missing: {missing_keys[:5]}"
        )



def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    enable_text_encoder=True,
    enable_vision_encoder=True,
    text_encoder_type=None,
    text_encoder_context_length=77,
    cache_dir=None,
    token=None,
    force_download=False,
    **kwargs,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path or Hugging Face identifier to model checkpoint
        load_from_HF: Whether to download checkpoint from Hugging Face if checkpoint_path is None
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: To enable compilation, set to True
        enable_text_encoder: Whether to enable text encoder
        enable_vision_encoder: Whether to enable vision encoder
        text_encoder_type: Optional student text encoder type for LiteText models
            (e.g. 'MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP2-L').
            If None, uses the standard SAM3 text encoder.
        text_encoder_context_length: Target context length for text encoder (default: 77).
            Only used when text_encoder_type is set. Common values: 16, 32, 77.
        cache_dir: Optional HF cache directory.
        token: Optional HF authentication token.
        force_download: Whether to force re-download from HF even if cached.

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    # Create visual components
    compile_mode = "default" if compile else None
    if enable_vision_encoder:
        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
        )
    else:
        vision_encoder = None

    # Create text components
    if enable_text_encoder:
        if text_encoder_type:
            # LiteText: init at ctx=77 to match checkpoint pos-embed, then truncate after load.
            text_encoder = _create_student_text_encoder(
                bpe_path, text_encoder_type, context_length=77
            )
        else:
            text_encoder = _create_text_encoder(bpe_path)
    else:
        text_encoder = None

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if checkpoint_path is not None:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    elif load_from_HF:
        checkpoint_path = download_ckpt_from_hf(
            repo_id=DEFAULT_SAM3_REPO,
            filename=DEFAULT_SAM3_CKPT,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, cache_dir=cache_dir, token=token, force_download=force_download)

    # Truncate text encoder context length after checkpoint loading
    if text_encoder_type and text_encoder_context_length < 77:
        model.backbone.language_backbone.set_context_length(text_encoder_context_length)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


import torch.nn.functional as F

class ImageStudentEncoder(nn.Module):
    def __init__(self, backbone, in_channels, embed_dim, embed_size, img_size):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.head(feats)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats

def _create_student_vision_backbone(
    backbone_type, model_name, compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create EfficientSAM3 visual backbone with a student backbone and neck."""
    
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    
    if backbone_type == "sam3":
        return _create_vision_backbone(
            compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
        )
    
    if backbone_type == "efficientvit":
        from sam3.backbones.efficientvit.efficientvit.backbone import (
            efficientvit_backbone_b0,
            efficientvit_backbone_b1,
            efficientvit_backbone_b2,
        )
        if model_name == "b0":
            backbone = efficientvit_backbone_b0()
        elif model_name == "b1":
            backbone = efficientvit_backbone_b1()
        elif model_name == "b2":
            backbone = efficientvit_backbone_b2()
        else:
            raise ValueError(f"Unknown EfficientViT model: {model_name}")
        
        class EfficientViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.channel_list = [model.width_list[-1]]
            
            def forward(self, x):
                x = x[0] if isinstance(x, list) else x
                out = self.model(x)
                return out['stage_final']
        
        wrapped_backbone = EfficientViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    elif backbone_type == "repvit":
        from sam3.backbones.repvit import (
            repvit_m0_9, repvit_m1_1, repvit_m2_3
        )
        name_map = {
            "m0.9": repvit_m0_9, "m0_9": repvit_m0_9,
            "m1.1": repvit_m1_1, "m1_1": repvit_m1_1,
            "m2.3": repvit_m2_3, "m2_3": repvit_m2_3,
        }
        if model_name not in name_map:
             raise ValueError(f"Unknown RepViT model: {model_name}")
        
        backbone = name_map[model_name](distillation=False)

        class RepViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Strip unused ImageNet classifier head (and distill head) so
                # the state_dict no longer references parameters that the
                # SAM3 checkpoint will never carry.
                for _attr in ("classifier", "distill_head"):
                    if hasattr(model, _attr):
                        delattr(model, _attr)
                # Infer channels via a dummy forward pass over the feature trunk.
                dummy = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    for f in model.features:
                        dummy = f(dummy)
                self.channel_list = [dummy.shape[1]]

            def forward(self, x):
                for f in self.model.features:
                    x = f(x)
                return x

        wrapped_backbone = RepViTTrunkWrapper(backbone)
        in_channels = wrapped_backbone.channel_list[0]

    elif backbone_type == "tinyvit":
        from sam3.backbones.tiny_vit import (
            tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
        )
        name_map = {
            "5m": tiny_vit_5m_224,
            "11m": tiny_vit_11m_224,
            "21m": tiny_vit_21m_224,
        }
        if model_name not in name_map:
             raise ValueError(f"Unknown TinyViT model: {model_name}")
        
        # num_classes=0 disables the unused ImageNet head so the state_dict
        # has no norm_head/head parameters that the SAM3 checkpoint would
        # never carry.
        backbone = name_map[model_name](img_size=1008, num_classes=0)

        class TinyViTTrunkWrapper(nn.Module):
            def __init__(self, model, channel_count):
                super().__init__()
                self.model = model
                self.channel_list = [channel_count]

            def forward(self, x):
                x = self.model.patch_embed(x)
                for layer in self.model.layers:
                    x = layer(x)
                # Reshape from (B, L, C) to (B, C, H, W)
                B, L, C = x.shape
                # Dynamic reshape assuming square
                side = int(L ** 0.5)
                x = x.view(B, side, side, C).permute(0, 3, 1, 2).contiguous()
                return x

        wrapped_backbone = TinyViTTrunkWrapper(
            backbone, backbone.layers[-1].dim
        )
        in_channels = wrapped_backbone.channel_list[0]

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # Wrap with ImageStudentEncoder to include the projection head
    student_encoder = ImageStudentEncoder(
        backbone=wrapped_backbone,
        in_channels=in_channels,
        embed_dim=1024, # SAM3 expects 1024 channels
        embed_size=72,
        img_size=1008,
    )
    
    # Add channel_list to student_encoder so Sam3DualViTDetNeck can read it
    student_encoder.channel_list = [1024]

    # Wrap student_encoder to return a list as expected by Sam3DualViTDetNeck
    class ListWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.channel_list = model.channel_list
            
        def forward(self, x):
            return [self.model(x)]
            
    final_trunk = ListWrapper(student_encoder)

    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        final_trunk,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def build_efficientsam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    backbone_type="efficientvit",
    model_name="b0",
    # Legacy argument support
    efficientvit_model=None,
    text_encoder_type=None, # e.g. "MobileCLIP-S0"
    text_encoder_context_length=77,
    cache_dir=None,
    token=None,
    force_download=False,
    **kwargs,
):
    """
    Build EfficientSAM3 image model with a student backbone

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path or Hugging Face identifier to EfficientSAM3 model checkpoint
        load_from_HF: Whether to load checkpoint from Hugging Face if checkpoint_path is None
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: To enable compilation, set to True
        backbone_type: Type of backbone ('efficientvit', 'repvit', 'tinyvit')
        model_name: Model variant (e.g. 'b0', 'm1.1', '5m')
        efficientvit_model: Deprecated, use backbone_type and model_name instead
        text_encoder_type: Type of text encoder (e.g. 'MobileCLIP-S0'). If None, uses standard SAM3 text encoder.
        text_encoder_context_length: Target context length for text encoder (default: 77).
            Only used when text_encoder_type is set. Common values: 16, 32, 77.
        cache_dir: Optional HF cache directory.
        token: Optional HF authentication token.
        force_download: Whether to force re-download from HF even if cached.

    Returns:
        An EfficientSAM3 image model
    """
    if efficientvit_model is not None:
        backbone_type = "efficientvit"
        model_name = efficientvit_model

    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    # Create visual components with student backbone
    compile_mode = "default" if compile else None
    vision_encoder = _create_student_vision_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    # Create text components
    if text_encoder_type:
        # LiteText: init at ctx=77 to match checkpoint pos-embed, then truncate after load.
        text_encoder = _create_student_text_encoder(bpe_path, text_encoder_type, context_length=77)
    else:
        text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if checkpoint_path is not None:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path,
            backbone_type=backbone_type,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    elif load_from_HF:
        checkpoint_path = download_ckpt_from_hf(
            repo_id=DEFAULT_EFFICIENTSAM3_REPO,
            backbone_type=backbone_type,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )

    # Truncate text encoder context length BEFORE checkpoint loading
    # This resizes positional embeddings so checkpoint can be loaded
    if text_encoder_type and text_encoder_context_length < 77:
        model.backbone.language_backbone.set_context_length(text_encoder_context_length)
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            checkpoint_path,
            backbone_type=backbone_type,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
    text_encoder_type: Optional[str] = None,
    text_encoder_context_length: int = 77,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    **kwargs,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file.
            When text_encoder_type is set, this should be a fully-merged LiteText
            video checkpoint containing detector + tracker + student text encoder weights.
        bpe_path: Path to the BPE tokenizer file
        text_encoder_type: Optional student text encoder type for LiteText models
            (e.g. 'MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP2-L').
            If None, uses the standard SAM3 text encoder.
        text_encoder_context_length: Target context length for text encoder (default: 77).
            Only used when text_encoder_type is set. Common values: 16, 32, 77.

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint
    if text_encoder_type:
        # LiteText video workflow:
        # 1. Swap text encoder with student variant (init with ctx=77 for ckpt compat)
        student_text_enc = _create_student_text_encoder(
            bpe_path, text_encoder_type, context_length=77
        )
        model.detector.backbone.language_backbone = student_text_enc

        # 2. Load fully-merged LiteText video checkpoint (detector + tracker + student text encoder)
        if checkpoint_path is not None:
            checkpoint_path = resolve_checkpoint_path(
                checkpoint_path,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
            ckpt = _load_state_dict_from_path(checkpoint_path, cache_dir=cache_dir, token=token)
            # Clean keys: remove student_trunk. prefix
            cleaned = {}
            for k, v in ckpt.items():
                new_k = k.replace("student_trunk.", "")
                cleaned[new_k] = v

            missing_keys, unexpected_keys = model.load_state_dict(cleaned, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        # 3. Truncate context length
        if text_encoder_context_length < 77:
            model.detector.backbone.language_backbone.set_context_length(text_encoder_context_length)
    else:
        # Standard SAM3 video model loading
        if checkpoint_path is not None:
            checkpoint_path = resolve_checkpoint_path(
                checkpoint_path,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
        elif load_from_HF:
            checkpoint_path = download_ckpt_from_hf(
                repo_id=DEFAULT_SAM3_REPO,
                filename=DEFAULT_SAM3_CKPT,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
        if checkpoint_path is not None:
            with g_pathmgr.open(checkpoint_path, "rb") as f:
                try:
                    ckpt = torch.load(f, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(f, map_location="cpu")
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]

            missing_keys, unexpected_keys = model.load_state_dict(
                ckpt, strict=strict_state_dict_loading
            )
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )


def _load_state_dict_from_path(
    checkpoint_path: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Load a checkpoint dict from path and return the raw state_dict-like mapping.

    Supports checkpoints saved as:
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - state_dict
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path, cache_dir=cache_dir, token=token)
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(f, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type at {checkpoint_path}: {type(ckpt)}")


def build_efficientsam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF: bool = False,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    strict_state_dict_loading: bool = False,
    apply_temporal_disambiguation: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compile: bool = False,
    backbone_type: str = "repvit",
    model_name: str = "m1.1",
    text_encoder_type: Optional[str] = None,
    text_encoder_context_length: int = 77,
    enable_inst_interactivity: bool = True,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    **kwargs,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """Build EfficientSAM3 video model (main-branch implementation).

    This variant swaps the default SAM3 vision backbone with a student backbone
    (EfficientViT/RepViT/TinyViT) while keeping the same detector+tracker wrapper.
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)
    compile_mode = "default" if compile else None
    visual_neck = _create_student_vision_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    if text_encoder_type:
        # LiteText: initialize with context_length=77 for checkpoint weight compatibility
        text_encoder = _create_student_text_encoder(bpe_path, text_encoder_type, context_length=77)
    else:
        text_encoder = _create_text_encoder(bpe_path)

    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head(
        compile_mode=compile_mode
    )
    input_geometry_encoder = _create_geometry_encoder()

    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )
    model = Sam3VideoInferenceWithInstanceInteractivity(
        detector=detector,
        tracker=tracker,
        score_threshold_detection=0.5,
        assoc_iou_thresh=0.1,
        det_nms_thresh=0.1,
        new_det_thresh=0.7,
        hotstart_delay=15 if apply_temporal_disambiguation else 0,
        hotstart_unmatch_thresh=8 if apply_temporal_disambiguation else 0,
        hotstart_dup_thresh=8 if apply_temporal_disambiguation else 0,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        recondition_every_nth_frame=16 if apply_temporal_disambiguation else 0,
        masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile,
    )

    if checkpoint_path is not None:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path,
            backbone_type=backbone_type,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    elif load_from_HF:
        checkpoint_path = download_ckpt_from_hf(
            repo_id=DEFAULT_EFFICIENTSAM3_REPO,
            backbone_type=backbone_type,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(f, map_location="cpu")
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        cleaned_ckpt = {}
        for k, v in ckpt.items():
            new_k = k.replace("student_trunk.", "")

            cleaned_ckpt[new_k] = v

        if not any(
            k.startswith("detector.") or k.startswith("tracker.")
            for k in cleaned_ckpt.keys()
        ) and any(k.startswith("backbone.") for k in cleaned_ckpt.keys()):
            cleaned_ckpt = {f"detector.{k}": v for k, v in cleaned_ckpt.items()}

        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}")

    # Truncate text encoder context length after checkpoint loading
    if text_encoder_type and text_encoder_context_length < 77:
        model.detector.backbone.language_backbone.set_context_length(text_encoder_context_length)

    model.to(device=device)
    return model


def build_efficientsam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    """Build a video predictor that uses the EfficientSAM3 video model builder."""
    model_kwargs = dict(model_kwargs)
    model_kwargs["use_efficientsam3"] = True
    return Sam3VideoPredictorMultiGPU(*model_args, gpus_to_use=gpus_to_use, **model_kwargs)


def from_pretrained(
    pretrained_model_name_or_path: Optional[str] = None,
    *model_args,
    backbone_type: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    eval_mode: bool = True,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    is_video: bool = False,
    **kwargs,
) -> nn.Module:
    """Load a SAM3 or EfficientSAM3 model just like standard Hugging Face load from pretrained.

    Args:
        pretrained_model_name_or_path: Path to checkpoint, HF repo ID (e.g. 'Simon7108528/EfficientSAM3',
            'facebook/sam3'), HF URI ('hf://...'), or HF URL.
        backbone_type: Vision backbone architecture ('tinyvit', 'efficientvit', 'repvit', 'sam3').
        model_name: Backbone variant (e.g. '11m', 'b0', 'm1.1').
        device: Device to place the model on ('cuda' or 'cpu').
        eval_mode: Whether to set model to evaluation mode (default: True).
        cache_dir: Hugging Face cache directory ($HF_HOME/hub or ~/.cache/huggingface/hub).
        token: Optional Hugging Face auth token.
        force_download: Whether to force re-download even if cached.
        is_video: Whether to build a video tracking model instead of an image model.
        **kwargs: Additional arguments passed to builder functions.

    Returns:
        Loaded SAM3 or EfficientSAM3 model instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_ident = str(pretrained_model_name_or_path or "").lower()

    # Determine whether this is an EfficientSAM3 model or standard SAM3
    is_efficient = False
    if backbone_type and backbone_type.lower() in ("tinyvit", "efficientvit", "repvit"):
        is_efficient = True
    elif any(k in raw_ident for k in ["efficient", "tinyvit", "repvit", "simon7108528"]):
        is_efficient = True

    if is_efficient:
        if backbone_type is None:
            if "tinyvit" in raw_ident:
                backbone_type = "tinyvit"
                model_name = model_name or "11m"
            elif "repvit" in raw_ident:
                backbone_type = "repvit"
                model_name = model_name or "m1.1"
            else:
                backbone_type = "efficientvit"
                model_name = model_name or "b0"

        if is_video:
            return build_efficientsam3_video_model(
                checkpoint_path=pretrained_model_name_or_path,
                load_from_HF=True if pretrained_model_name_or_path is None else False,
                backbone_type=backbone_type,
                model_name=model_name or "m1.1",
                device=device,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                **kwargs,
            )
        else:
            return build_efficientsam3_image_model(
                checkpoint_path=pretrained_model_name_or_path,
                load_from_HF=True if pretrained_model_name_or_path is None else False,
                backbone_type=backbone_type,
                model_name=model_name or ("11m" if backbone_type == "tinyvit" else ("m1.1" if backbone_type == "repvit" else "b0")),
                device=device,
                eval_mode=eval_mode,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                **kwargs,
            )
    else:
        if is_video:
            return build_sam3_video_model(
                checkpoint_path=pretrained_model_name_or_path,
                load_from_HF=True,
                device=device,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                **kwargs,
            )
        else:
            return build_sam3_image_model(
                checkpoint_path=pretrained_model_name_or_path,
                load_from_HF=True,
                device=device,
                eval_mode=eval_mode,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                **kwargs,
            )


build_sam3_image_model.from_pretrained = from_pretrained
build_efficientsam3_image_model.from_pretrained = from_pretrained
build_sam3_video_model.from_pretrained = from_pretrained
build_efficientsam3_video_model.from_pretrained = from_pretrained
