#!/usr/bin/env python3
"""
Benchmark script for EfficientSAM3 (TinyViT student backbone + MobileCLIP-S0 text encoder)
Measures latency, GPU memory (VRAM), parameters, and evaluates segmentation on sample image.
"""

import time
import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS, plot_bbox, plot_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"EfficientSAM3 Local Benchmark on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"==================================================")

    ckpt_path = "checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt"
    image_path = "assets/dog_person.jpeg" if os.path.exists("assets/dog_person.jpeg") else "efficientsam3/sam3/assets/dog_person.jpeg"
    output_vis = "benchmark_vis.png"

    # Reset max memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_load = time.perf_counter()
    model = build_efficientsam3_image_model(
        checkpoint_path=ckpt_path,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
    )
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in: {load_time:.2f} seconds")

    # Parameter Breakdown
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.backbone.vision_backbone.parameters())
    text_params = sum(p.numel() for p in model.backbone.language_backbone.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    
    print("\n--- Model Architecture & Parameter Counts ---")
    print(f"  Vision Backbone (TinyViT-11M + Neck) : {vision_params / 1e6:6.2f} M params")
    print(f"  Text Encoder (MobileCLIP-S0)          : {text_params / 1e6:6.2f} M params")
    print(f"  Transformer (DETR Encoder/Decoder)    : {transformer_params / 1e6:6.2f} M params")
    print(f"  Total Model Parameters               : {total_params / 1e6:6.2f} M params")
    print(f"  (Comparison: Original SAM3 is ~861.5 M params -> ~88% reduction!)")

    processor = Sam3Processor(model)
    raw_image = Image.open(image_path).convert("RGB")

    # Warmup
    print("\nWarming up GPU with 3 passes...")
    for _ in range(3):
        state = processor.set_image(raw_image)
        _ = processor.set_text_prompt("person", state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark: Image Feature Extraction (Vision Backbone)
    print("\n--- Running Latency Benchmarks (averaged over 10 runs) ---")
    num_runs = 10
    vision_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = processor.set_image(raw_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vision_times.append(time.perf_counter() - t0)

    avg_vision_ms = np.mean(vision_times) * 1000
    print(f"  Image Encoding (Vision Backbone) : {avg_vision_ms:6.2f} ms")

    # Benchmark: Prompt & Segmentation Head
    prompt_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = processor.set_text_prompt("dog", state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prompt_times.append(time.perf_counter() - t0)

    avg_prompt_ms = np.mean(prompt_times) * 1000
    print(f"  Text Prompt + Mask Prediction    : {avg_prompt_ms:6.2f} ms")

    total_latency_ms = avg_vision_ms + avg_prompt_ms
    fps = 1000.0 / total_latency_ms
    print(f"  End-to-End Latency               : {total_latency_ms:6.2f} ms ({fps:.1f} FPS)")

    # VRAM Stats
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        print("\n--- GPU Memory Footprint ---")
        print(f"  Peak VRAM Allocated: {peak_vram_mb:6.1f} MB (~{peak_vram_mb/1024:.2f} GB)")
        print(f"  Peak VRAM Reserved : {peak_res_mb:6.1f} MB (~{peak_res_mb/1024:.2f} GB)")

    # Qualitative Test: Segment 'dog' and 'person'
    print("\n--- Qualitative Test on dog_person.jpeg ---")
    test_prompts = ["dog", "person"]
    fig, axes = plt.subplots(1, len(test_prompts), figsize=(14, 7))

    for idx, prompt in enumerate(test_prompts):
        ax = axes[idx]
        inference_state = processor.set_image(raw_image)
        inference_state = processor.set_text_prompt(prompt, inference_state)

        masks = inference_state["masks"]
        scores = inference_state["scores"]
        boxes = inference_state["boxes"]
        num_masks = int(masks.shape[0])
        print(f"  Prompt '{prompt}': detected {num_masks} instance(s), confidence scores: {[round(float(s), 3) for s in scores]}")

        ax.imshow(raw_image)
        w, h = raw_image.size
        for i in range(num_masks):
            color = COLORS[i % len(COLORS)]
            # Draw mask
            mask_np = masks[i].squeeze(0).cpu().numpy().astype(bool)
            img_rgba = np.zeros((h, w, 4), dtype=np.float32)
            c = np.array(color)
            if c.max() > 1.0:
                c = c / 255.0
            img_rgba[mask_np] = [c[0], c[1], c[2], 0.6]
            ax.imshow(img_rgba)

            prob = float(scores[i].item())
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c[:3], linewidth=2.5)
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 8), f"{prompt}: {prob:.2f}", color="white", fontsize=11, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=c[:3], alpha=0.85))

        ax.set_title(f"Prompt: '{prompt}' ({num_masks} found)", fontsize=13, weight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_vis, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nVisualization saved to: {output_vis}")
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
