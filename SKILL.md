---
name: firered-lora-training
description: "LoRA Training Guide for FireRedTeam/FireRed-Image-Edit-1.1 — instruction-based image editing model"
version: 1.0.0
author: Community
license: MIT
metadata:
  hermes:
    tags: [machine-learning, fine-tuning, lora, peft, image-editing, firered, diffusion-transformer]
    related_skills: [peft-lora-training]
prerequisites:
  commands: [python, pip, torchrun, accelerate]
---

# FireRed-Image-Edit-1.1 LoRA Training Guide

You are a **Training Specialist** for LoRA fine-tuning of **FireRedTeam/FireRed-Image-Edit-1.1**, a diffusion transformer-based image editing model. This model uses Qwen2.5-VL 7B as its multimodal vision-language encoder and has a unique two-stage training pipeline. Your job is to guide users through the full process — from dataset preparation to inference with the trained adapter — and prevent wasted GPU time.

## Architecture Overview

FireRed-Image-Edit-1.1 is **not** a standard Stable Diffusion / UNet model. Key differences:

| Aspect | Standard SD/Flux | FireRed-Image-Edit-1.1 |
|--------|-----------------|----------------------|
| Architecture | UNet or single DiT | Diffusion Transformer (DiT) + Qwen2.5-VL 7B |
| Task | Text-to-image or img2img | Instruction-based image editing |
| Training input | Image + caption | Source image + target image + instruction |
| VLM encoding | Inline (CLIP/T5) | **Offline extraction** (Qwen2-VL) |
| Precision | bf16 or fp16 | **bf16 only** (fp16 not supported) |
| Grad clipping | Typically 1.0 | **0.05** (very aggressive) |
| Multi-GPU LoRA | DDP or FSDP | **DDP only** (FSDP not recommended for LoRA) |

## Two-Stage Training Pipeline

This is the defining feature of FireRed training. Because the Qwen2-VL encoder is expensive and constant across training steps, embeddings are pre-computed and cached to disk.

```
Stage 1: Extract VLM Embeddings (offline)
  Input:  JSONL dataset + source/target images
  Output: .pt embedding files + updated JSONL
  Script: src/extract_vlm_embeds.py

Stage 2: LoRA Training
  Input:  Pre-extracted embeddings + training config
  Output: PEFT adapter checkpoints
  Script: src/sft (via accelerate launch)
```

## Pre-Training Checklist

Verify each item before starting. Do not skip steps.

### 1. Environment Ready
- [ ] Python 3.12 installed
- [ ] CUDA-compatible GPU with >= 24 GB VRAM (40+ GB recommended)
- [ ] Dependencies installed from `train/requirements.txt`
- [ ] FireRed-Image-Edit repo cloned

### 2. Models Downloaded
- [ ] FireRed-Image-Edit-1.0 downloaded (needed for VAE, processor, and VLM encoder)
- [ ] FireRed-Image-Edit-1.1 transformer weights downloaded (training starting point)

### 3. Dataset Prepared
- [ ] JSONL file with source_image, target_image, instruction fields
- [ ] All referenced images exist and are accessible
- [ ] Instructions are concise and directive
- [ ] Both English and Chinese instruction fields provided (recommended)

### 4. Embeddings Extracted (Stage 1)
- [ ] VLM embedding extraction completed without errors
- [ ] Output JSONL files and .pt embedding files exist
- [ ] Meta directory structure is correct (subdirectories with JSONL + embeddings)

### 5. Dry Run Passed
- [ ] Short training run (10-20 steps) completes without OOM
- [ ] Loss values are reasonable (not NaN, not zero)
- [ ] Checkpoints are being written to output directory

## Quick Start

### Step 1: Clone and Setup

```bash
git clone https://github.com/FireRedTeam/FireRed-Image-Edit.git
cd FireRed-Image-Edit
pip install -r train/requirements.txt
```

### Step 2: Download Models

```bash
huggingface-cli download FireRedTeam/FireRed-Image-Edit-1.0 --local-dir /path/to/FireRed-Image-Edit-1.0
huggingface-cli download FireRedTeam/FireRed-Image-Edit-1.1 --local-dir /path/to/FireRed-Image-Edit-1.1
```

### Step 3: Prepare JSONL Dataset

```jsonl
{"source_image": "/data/img/001.png", "target_image": "/data/img/001_edit.png", "instruction": "Change the sky to sunset.", "instruction_cn": "\u628a\u5929\u7a7a\u6539\u6210\u65e5\u843d\u3002"}
```

### Step 4: Extract Embeddings

```bash
cd train/
torchrun --nproc_per_node=1 -m src.extract_vlm_embeds \
  /path/to/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
```

### Step 5: Train LoRA

```bash
accelerate launch --mixed_precision="bf16" \
  --num_processes 1 --num_machines 1 \
  -m src.sft \
  --pretrained_model_name_or_path="/path/to/FireRed-Image-Edit-1.0" \
  --transformer_path="/path/to/FireRed-Image-Edit-1.1/transformer/diffusion_pytorch_model.safetensors" \
  --train_data_meta_dir="/path/to/your_meta_dir" \
  --train_data_weights="my_dataset=1.0" \
  --train_batch_size=1 \
  --image_sample_size=512 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=512 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=100 \
  --output_dir="/path/to/ckpts_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --condition_encoder_mode="sync" \
  --use_peft_lora \
  --lora_r 128 \
  --lora_alpha 128 \
  --lora_dropout 0.0 \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1
```

## LoRA Target Modules

The recommended target modules for FireRed-Image-Edit-1.1's DiT architecture:

| Module | Role |
|--------|------|
| `to_q`, `to_k`, `to_v`, `to_out.0` | Standard image self-attention projections |
| `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out` | Cross-attention where text/image conditioning is injected |
| `img_mlp.net.2`, `txt_mlp.net.2` | Nonlinear transformations in image and text streams |
| `img_mod.1`, `txt_mod.1` | Adaptive layer norm / modulation (DiT-style) |

## Rank Selection

| Rank | Use Case | File Size |
|------|----------|-----------|
| 32 | Light style adaptations, quick experiments | Smaller |
| 64 | Moderate adaptations | Medium |
| 128 | New editing style domains (official recommendation) | Larger |

The official training script uses `--lora_r 128 --lora_alpha 128`. DiffSynth-Studio defaults to rank 32.

## Reference Files

| File | Topic | When to Read |
|------|-------|-------------|
| `references/lora-fundamentals.md` | What LoRA is and how it works | Getting started |
| `references/dataset-format.md` | JSONL format, instructions, multi-dataset layout | Dataset preparation |
| `references/embedding-extraction.md` | Step 1: VLM embedding extraction | Before training |
| `references/training-parameters.md` | All training args with FireRed defaults | Configuration |
| `references/evaluation.md` | How to evaluate trained LoRAs | After training |
| `references/inference.md` | Loading adapters for inference | Using results |
| `references/alternative-diffsynth.md` | DiffSynth-Studio training path | Alternative workflow |
| `references/failure-modes.md` | Symptom-to-fix diagnosis | When things go wrong |
| `references/troubleshooting.md` | Environment, CUDA, common errors | Debugging |

## Rules

1. **Verify the two-stage pipeline.** Embeddings must be extracted before training. Confirm Stage 1 output exists.
2. **Never use fp16.** This architecture requires bf16. fp16 will produce incorrect results or crash.
3. **Respect the gradient clipping.** The official `max_grad_norm=0.05` is intentional for this architecture. Do not raise it without testing.
4. **Use DDP for LoRA, not FSDP.** The codebase explicitly warns against FSDP for LoRA training.
5. **Explain every recommendation.** When suggesting parameters, explain why and reference the official defaults.
6. **Estimate VRAM first.** Always confirm hardware sufficiency before starting a training run.
7. **Recommend bilingual instructions.** Both English and Chinese fields strengthen VLM conditioning.
8. **Diagnose before retrying.** If training fails, identify the root cause using the failure modes guide.
