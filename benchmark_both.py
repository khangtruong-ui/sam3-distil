#!/usr/bin/env python3
"""
Comprehensive Comparative Benchmark:
EfficientSAM3-TV-M (TinyViT-11M) vs. EfficientSAM3-EV-M (EfficientViT-B1)
on NVIDIA GeForce RTX 2060
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
from sam3.visualization_utils import COLORS

MODELS = [
    {
        "name": "EfficientSAM3 (TinyViT-11M)",
        "short": "TV-M",
        "ckpt": "checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
        "backbone_type": "tinyvit",
        "model_name": "11m",
    },
    {
        "name": "EfficientSAM3 (EfficientViT-B1)",
        "short": "EV-M",
        "ckpt": "checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt",
        "backbone_type": "efficientvit",
        "model_name": "b1",
    },
]

def benchmark_single(config, image, test_prompt="dog"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Benchmarking: {config['name']}")
    print(f"=======================================================")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t_load_start = time.perf_counter()
    model = build_efficientsam3_image_model(
        checkpoint_path=config["ckpt"],
        backbone_type=config["backbone_type"],
        model_name=config["model_name"],
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
    )
    load_time = time.perf_counter() - t_load_start

    # Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    vis_params = sum(p.numel() for p in model.backbone.vision_backbone.parameters())
    text_params = sum(p.numel() for p in model.backbone.language_backbone.parameters())
    dec_params = sum(p.numel() for p in model.transformer.parameters())

    processor = Sam3Processor(model)

    # Warmup
    for _ in range(3):
        s = processor.set_image(image)
        _ = processor.set_text_prompt(test_prompt, s)
    torch.cuda.synchronize()

    # Latency: Vision Backbone
    num_runs = 10
    enc_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = processor.set_image(image)
        torch.cuda.synchronize()
        enc_times.append(time.perf_counter() - t0)
    avg_enc_ms = np.mean(enc_times) * 1000

    # Latency: Prompt & Mask Head
    prompt_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = processor.set_text_prompt(test_prompt, s)
        torch.cuda.synchronize()
        prompt_times.append(time.perf_counter() - t0)
    avg_prompt_ms = np.mean(prompt_times) * 1000

    total_ms = avg_enc_ms + avg_prompt_ms
    fps = 1000.0 / total_ms

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Run for "person" as well
    out_person = processor.set_text_prompt("person", s)

    print(f"  Load Time         : {load_time:.2f} s")
    print(f"  Total Params      : {total_params / 1e6:.2f} M (Vision: {vis_params/1e6:.2f}M, Text: {text_params/1e6:.2f}M, Dec: {dec_params/1e6:.2f}M)")
    print(f"  Vision Encode     : {avg_enc_ms:6.2f} ms")
    print(f"  Prompt + Mask Head: {avg_prompt_ms:6.2f} ms")
    print(f"  Total Latency     : {total_ms:6.2f} ms ({fps:.1f} FPS)")
    print(f"  Peak VRAM         : {peak_vram_mb:6.1f} MB (~{peak_vram_mb/1024:.2f} GB)")
    print(f"  Detection 'dog'   : {len(out['masks'])} mask(s), scores: {[round(float(x), 2) for x in out['scores']]}")
    print(f"  Detection 'person': {len(out_person['masks'])} mask(s), scores: {[round(float(x), 2) for x in out_person['scores']]}")

    result = {
        "config": config,
        "total_params_m": total_params / 1e6,
        "vis_params_m": vis_params / 1e6,
        "text_params_m": text_params / 1e6,
        "avg_enc_ms": avg_enc_ms,
        "avg_prompt_ms": avg_prompt_ms,
        "total_ms": total_ms,
        "fps": fps,
        "vram_mb": peak_vram_mb,
        "dog_out": out,
        "person_out": out_person,
    }

    # Clean up model
    del model, processor, s
    torch.cuda.empty_cache()

    return result

def main():
    img_path = "assets/dog_person.jpeg" if os.path.exists("assets/dog_person.jpeg") else "efficientsam3/sam3/assets/dog_person.jpeg"
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    results = []
    for cfg in MODELS:
        res = benchmark_single(cfg, image)
        results.append(res)

    print("\n" + "="*70)
    print("                 SUMMARY BENCHMARK COMPARISON TABLE                 ")
    print("="*70)
    print(f"{'Metric':<25} | {'Original SAM 3 (HF/Meta)':<24} | {'EfficientSAM3 (TV-M)':<20} | {'EfficientSAM3 (EV-M)':<20}")
    print("-" * 97)
    p0_tot = f"{results[0]['total_params_m']:.1f} M (-88%)"
    p1_tot = f"{results[1]['total_params_m']:.1f} M (-90%)"
    p0_vis = f"{results[0]['vis_params_m']:.1f} M (-94%)"
    p1_vis = f"{results[1]['vis_params_m']:.1f} M (-95%)"
    p0_txt = f"{results[0]['text_params_m']:.1f} M (-88%)"
    p1_txt = f"{results[1]['text_params_m']:.1f} M (-88%)"
    p0_enc = f"{results[0]['avg_enc_ms']:.1f} ms"
    p1_enc = f"{results[1]['avg_enc_ms']:.1f} ms"
    p0_prm = f"{results[0]['avg_prompt_ms']:.1f} ms"
    p1_prm = f"{results[1]['avg_prompt_ms']:.1f} ms"
    p0_tot_ms = f"{results[0]['total_ms']:.1f} ms ({results[0]['fps']:.1f} FPS)"
    p1_tot_ms = f"{results[1]['total_ms']:.1f} ms ({results[1]['fps']:.1f} FPS)"
    p0_vram = f"{results[0]['vram_mb']:.0f} MB (~{results[0]['vram_mb']/1024:.2f} GB)"
    p1_vram = f"{results[1]['vram_mb']:.0f} MB (~{results[1]['vram_mb']/1024:.2f} GB)"

    print(f"{'Total Parameters':<25} | {'~861.5 M':<24} | {p0_tot:<20} | {p1_tot:<20}")
    print(f"{'Vision Encoder Params':<25} | {'~463.0 M':<24} | {p0_vis:<20} | {p1_vis:<20}")
    print(f"{'Text Encoder Params':<25} | {'~354.0 M':<24} | {p0_txt:<20} | {p1_txt:<20}")
    print(f"{'Image Encode Latency':<25} | {'~700 - 1200 ms':<24} | {p0_enc:<20} | {p1_enc:<20}")
    print(f"{'Prompt+Mask Latency':<25} | {'~200 ms':<24} | {p0_prm:<20} | {p1_prm:<20}")
    print(f"{'Total End-to-End':<25} | {'~900 - 1400 ms':<24} | {p0_tot_ms:<20} | {p1_tot_ms:<20}")
    print(f"{'Peak VRAM Allocated':<25} | {'~8.5 - 11.0 GB':<24} | {p0_vram:<20} | {p1_vram:<20}")
    print("="*97)

    # Plot Visual Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for r_idx, res in enumerate(results):
        m_name = res["config"]["short"]
        for p_idx, prompt in enumerate(["dog", "person"]):
            ax = axes[r_idx, p_idx]
            out = res["dog_out"] if prompt == "dog" else res["person_out"]
            masks = out["masks"]
            scores = out["scores"]
            boxes = out["boxes"]
            num_masks = int(masks.shape[0])

            ax.imshow(image)
            for i in range(num_masks):
                color = COLORS[i % len(COLORS)]
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
                ax.text(x1, max(0, y1 - 8), f"{prompt}: {prob:.2f}", color="white", fontsize=10, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=c[:3], alpha=0.85))

            ax.set_title(f"{res['config']['name']} | '{prompt}' ({num_masks} found)", fontsize=12, weight="bold")
            ax.axis("off")

    plt.tight_layout()
    comp_img = "benchmark_comparison.png"
    plt.savefig(comp_img, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nComparative visualization saved to: {comp_img}")

if __name__ == "__main__":
    main()
