# Alternative: DiffSynth-Studio Training Path

## Overview

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope provides an alternative LoRA training pipeline for FireRed-Image-Edit-1.1. It uses a unified `train.py` entry point with a different configuration approach.

## Key Differences from Official Pipeline

| Aspect | Official FireRedTeam | DiffSynth-Studio |
|--------|---------------------|------------------|
| Dataset format | JSONL | Metadata JSON |
| Embedding extraction | Separate Step 1 | Handled internally |
| Model specification | CLI path args | Inline `model_id:glob` strings |
| Default rank | 128 | 32 |
| Default learning rate | 2e-5 | 1e-4 (5x higher) |
| LoRA target | `--lora_target_modules` | `--lora_target_modules` + `--lora_base_model "dit"` |
| Checkpoint naming | `checkpoint-{step}/` | `epoch-{n}.safetensors` |

## Training Command

```bash
# Download example dataset
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "qwen_image/FireRed-Image-Edit-1.1/*" \
  --local_dir ./data/diffsynth_example_dataset

# Train
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/FireRed-Image-Edit-1.1 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/FireRed-Image-Edit-1.1/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "FireRedTeam/FireRed-Image-Edit-1.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FireRed-Image-Edit-1.1_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

## Arguments Explained

| Argument | Description |
|----------|-------------|
| `--dataset_base_path` | Root directory containing images |
| `--dataset_metadata_path` | JSON file describing the dataset |
| `--data_file_keys` | Which keys in metadata map to image files |
| `--extra_inputs` | Additional inputs (the edit target image) |
| `--max_pixels` | Maximum pixel count (1048576 = 1024x1024) |
| `--dataset_repeat` | How many times to repeat the dataset per epoch |
| `--model_id_with_origin_paths` | Model components as `model_id:glob_pattern` strings |
| `--remove_prefix_in_ckpt` | Strip pipeline prefix for standalone adapter use |
| `--lora_base_model` | Which model component to apply LoRA to (`"dit"`) |
| `--lora_target_modules` | Same target modules as official pipeline |
| `--lora_rank` | LoRA rank (default 32 vs official 128) |
| `--find_unused_parameters` | Required for DDP with LoRA |

## When to Use DiffSynth-Studio

**Choose DiffSynth-Studio when:**
- You want a simpler setup without the two-stage extraction pipeline
- You're already using DiffSynth-Studio for other models
- You want quick experimentation with lower rank (32)
- You prefer metadata JSON over JSONL format

**Choose the official FireRedTeam pipeline when:**
- You want maximum control over the training process
- You need high-rank LoRA (128) for complex domain adaptation
- You're training on large datasets where offline embedding extraction saves significant time
- You want to follow the exact recipe used for the official LoRA Zoo models

## Inference After DiffSynth Training

See `references/inference.md` — Method 2 covers DiffSynth-Studio inference with the `QwenImagePipeline`.

## Important Notes

- The target modules are **identical** between both pipelines
- DiffSynth uses a **5x higher default learning rate** (1e-4 vs 2e-5). If you're porting a recipe between pipelines, adjust accordingly.
- DiffSynth checkpoints save as `epoch-{n}.safetensors` rather than PEFT adapter directories. Loading differs — see inference guide.
