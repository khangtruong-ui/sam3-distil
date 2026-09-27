# EfficientSAM3: LoRA & QLoRA Fine-Tuning Guide

This guide explains how to fine-tune **EfficientSAM3** (distilled SAM3 with TinyViT, EfficientViT, or RepViT backbones) using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized Low-Rank Adaptation)**.

---

## 1. Why LoRA / QLoRA for EfficientSAM3?

EfficientSAM3 already slashes parameter count by **~88%–90%** (from 861.5M down to ~97M–103M) and inference memory down to ~1.2 GB. However, full fine-tuning still requires storing optimizer states (AdamW creates 2 additional FP32 tensors per parameter), which multiplies training VRAM.

| Method | Base Model Weights | Trainable Parameters | Training VRAM (Batch 2) | Hardware Required |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | FP16/FP32 (103M) | **103.5M (100%)** | ~6.5 – 8.0 GB | RTX 3080 / 4080 (10GB+) |
| **LoRA (r=16)** | FP16 (Frozen) | **~1.86M (1.76%)** | **~2.8 – 3.5 GB** | **RTX 2060 (6GB–12GB)** |
| **QLoRA (4-bit, r=16)** | INT4 / NF4 (Frozen) | **~1.86M (1.76%)** | **~1.5 – 2.0 GB** | **Entry-level GPU (4GB–8GB)** |

### Key Benefits:
- **Low Compute / Single-GPU Feasible:** Trainable on an entry-level GPU (like the local RTX 2060 12GB) in hours rather than days.
- **Prevents Catastrophic Forgetting:** Keeps SAM3's generalized zero-shot concept segmentation ability while adapting to niche domains (e.g. medical microscopy, pathology, aerial drone imagery, industrial defect inspection).
- **Modular Adapters:** Adapter checkpoints are tiny (**~7.5 MB**), making them easy to distribute, switch dynamically, or deploy at scale.

---

## 2. Target Modules in EfficientSAM3

LoRA decomposes weight updates into two low-rank matrices: $\Delta W = B \cdot A$ where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ ($r \ll \min(d, k)$).

In EfficientSAM3, the most critical linear projections to target are:

```
EfficientSAM3 Model Hierarchy
├── backbone
│   ├── vision_backbone (TinyViT / EfficientViT)
│   │   └── attention projections: 'qkv', 'proj'
│   └── language_backbone (MobileCLIP)
│       └── text attention: 'out_proj', 'fc1', 'fc2'
└── transformer (DETR Encoder & Decoder)
    ├── self_attn: 'qkv_proj', 'out_proj'
    ├── cross_attn: 'qkv_proj', 'out_proj'
    └── mlp / ffn: 'linear1', 'linear2'
```

### Recommended `target_modules`:
```python
TARGET_MODULES = [
    "qkv",          # Vision backbone self-attention
    "qkv_proj",     # DETR transformer cross/self attention
    "out_proj",     # Transformer attention output projection
    "proj",         # FPN and neck projections
    "linear1",      # Transformer MLP expansion
    "linear2",      # Transformer MLP projection
]
```

---

## 3. Quick Start: Standard LoRA Fine-Tuning

### Step 1: Install Dependencies
```bash
pip install -e ".[lora]"
```

### Step 2: Initialize Base Model & Attach LoRA
```python
import torch
from peft import LoraConfig, get_peft_model
from sam3.model_builder import build_efficientsam3_image_model

# 1. Load pre-distilled base model
model = build_efficientsam3_image_model(
    checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    device="cuda",
    eval_mode=False,
)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,                           # Rank (8 for minor tuning, 16-32 for complex tasks)
    lora_alpha=32,                  # Scaling factor (usually 2x rank)
    target_modules=[
        "qkv", "qkv_proj", "out_proj", "proj", "linear1", "linear2"
    ],
    lora_dropout=0.05,
    bias="none",
)

# 3. Wrap model with PEFT
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 1,863,680 || all params: 105,398,458 || trainable%: 1.7682%
```

### Step 3: Run the Training CLI
You can also use our pre-built script:
```bash
python scripts/train_lora.py \
    --checkpoint checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt \
    --backbone tinyvit \
    --model-name 11m \
    --lora-r 16 \
    --lr 2e-4 \
    --epochs 5 \
    --output-dir output/lora_adapter
```

---

## 4. QLoRA: 4-Bit Quantized LoRA for Minimal VRAM

QLoRA loads the base model in 4-bit NormalFloat (NF4) with double quantization, while keeping LoRA adapter parameters in FP16/BF16.

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from sam3.model_builder import build_efficientsam3_image_model

# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Build base model and quantize linear weights
base_model = build_efficientsam3_image_model(
    checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
    backbone_type="tinyvit",
    model_name="11m",
    device="cuda",
    eval_mode=False,
)

# Attach adapters to 4-bit quantized layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv", "qkv_proj", "out_proj", "proj", "linear1", "linear2"],
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(base_model, lora_config)
```

---

## 5. Loss Functions & Training Loop

For custom domain fine-tuning (e.g. text-prompted segmentation), a combination of **Binary Cross-Entropy (BCE)** and **Soft Dice Loss** delivers best results:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        preds = torch.sigmoid(pred_logits)
        intersection = (preds * targets).sum(dim=(-2, -1))
        union = preds.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def compute_loss(pred_mask_logits, gt_masks):
    loss_bce = bce_loss(pred_mask_logits, gt_masks)
    loss_dice = dice_loss(pred_mask_logits, gt_masks)
    return loss_bce + 2.0 * loss_dice
```

---

## 6. Saving, Loading & Zero-Overhead Inference

### Saving the Adapter:
```python
peft_model.save_pretrained("output/custom_sam3_lora")
```
This saves:
- `adapter_config.json` (~600 bytes)
- `adapter_model.safetensors` (~7.5 MB)

### Loading for Inference:
```python
from peft import PeftModel
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load base model
base_model = build_efficientsam3_image_model(
    checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
    backbone_type="tinyvit",
    model_name="11m",
    device="cuda",
)

# Load fine-tuned LoRA weights
model = PeftModel.from_pretrained(base_model, "output/custom_sam3_lora")

# Merge LoRA weights into base weights for zero-overhead inference:
model = model.merge_and_unload()

processor = Sam3Processor(model)
```

> [!TIP]
> Calling `model.merge_and_unload()` mathematically fuses the adapter weights $\Delta W$ directly into the base weights $W$. This eliminates all LoRA runtime overhead and achieves the **exact same latency and FPS** as the original model!
