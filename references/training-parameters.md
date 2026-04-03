# Training Parameters

All parameters for FireRed-Image-Edit-1.1 LoRA training, with official defaults and recommendations.

## LoRA-Specific Parameters

| Argument | Default | Official Example | Description |
|----------|---------|-----------------|-------------|
| `--use_peft_lora` | off | **Required** | Flag to enable PEFT LoRA mode. Without this, full fine-tuning runs. |
| `--lora_r` | `32` | `128` | LoRA rank. Higher = more capacity. |
| `--lora_alpha` | `32` | `128` | Scaling factor. Typically set equal to rank. |
| `--lora_dropout` | `0.0` | `0.0` | Dropout on LoRA layers. Leave at 0.0 for image generation. |
| `--lora_target_modules` | (see below) | (see below) | Comma-separated module names to apply LoRA to. |
| `--lora_path` | `None` | — | Path to a saved adapter for resumed training. |

### Default Target Modules

```
to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1
```

### LoRA Initialization

The codebase uses `init_lora_weights="gaussian"` (Gaussian initialization, not the PEFT default kaiming). This is set internally in `src/model_provider.py` and cannot be changed via CLI.

## Data & Model Paths

| Argument | Description |
|----------|-------------|
| `--pretrained_model_name_or_path` | Path to FireRed-Image-Edit-1.0 model root (provides VAE and processor) |
| `--transformer_path` | Path to 1.1 transformer weights (starting point for LoRA training) |
| `--train_data_meta_dir` | Root directory whose subdirectories are training datasets |
| `--output_dir` | Directory to save checkpoints and adapter weights |

### Using 1.1 Transformer Weights

Always specify the 1.1 weights as your training starting point:

```
--pretrained_model_name_or_path="/path/to/FireRed-Image-Edit-1.0" \
--transformer_path="/path/to/FireRed-Image-Edit-1.1/transformer/diffusion_pytorch_model.safetensors"
```

The 1.0 model provides the VAE and processor. The 1.1 weights are the improved transformer you are fine-tuning.

## Optimization Parameters

| Argument | Official Default | Description |
|----------|-----------------|-------------|
| `--learning_rate` | `2e-05` | Initial LR after warmup |
| `--lr_scheduler` | `constant_with_warmup` | Options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` |
| `--lr_warmup_steps` | `100` | Number of warmup steps |
| `--adam_weight_decay` | `3e-2` | Weight decay — FireRed uses 0.03 (higher than typical 0.01) |
| `--adam_epsilon` | `1e-10` | Adam epsilon — tighter than the default 1e-8 |
| `--max_grad_norm` | `0.05` | **Very aggressive** gradient clipping. Do not raise without testing. |
| `--gradient_accumulation_steps` | `1` | Increase when reducing GPU count to maintain effective batch size |
| `--mixed_precision` | `bf16` | **bf16 is required.** fp16 is not supported. |

### Why Such Aggressive Gradient Clipping?

The official `max_grad_norm=0.05` is intentional for this DiT architecture. The modulation layers (`img_mod.1`, `txt_mod.1`) and cross-attention projections can produce large gradients early in training. The aggressive clipping prevents gradient explosions that would otherwise destabilize training.

**Do not raise this value** unless you have tested thoroughly and have a specific reason.

## Training Duration

| Argument | Description |
|----------|-------------|
| `--num_train_epochs` | Total epochs. Overridden by `--max_train_steps` if set. |
| `--max_train_steps` | Hard cap on total steps. Official example: `512`. |
| `--checkpointing_steps` | Save a checkpoint every N steps. Official example: `100`. |
| `--checkpoints_total_limit` | Maximum checkpoints to keep (oldest pruned). |
| `--resume_from_checkpoint` | Path to a checkpoint dir, or `"latest"` to auto-resume. |

### Recommended Training Duration

| Dataset Size | Steps | Notes |
|-------------|-------|-------|
| Small (50-200 pairs) | 256-512 | Good starting point, watch for overfitting |
| Medium (200-1000 pairs) | 512-2000 | Standard training |
| Large (1000+ pairs) | 2000-5000 | Extended training, save frequent checkpoints |

## Memory & Performance

| Argument | Description |
|----------|-------------|
| `--gradient_checkpointing` | **Highly recommended.** Reduces VRAM at cost of compute. |
| `--train_batch_size` | Per-device batch size. Official LoRA example: `2`. Use `1` for single GPU. |
| `--image_sample_size` | Training resolution (square). Default `512`. Higher uses more VRAM. |
| `--vae_mini_batch` | Mini-batch size for VAE encode/decode. Lower reduces VRAM. |
| `--dataloader_num_workers` | DataLoader workers. `0` = main process only. |
| `--allow_tf32` | Enable TF32 on Ampere GPUs for faster matmuls. |

## Data Weighting

| Argument | Format | Description |
|----------|--------|-------------|
| `--train_data_weights` | `"taskA=0.5,taskB=1.2"` | Sampling weight per subdirectory. Tasks not listed are excluded. |
| `--train_src_img_num_weights` | `"0=1,1=1,2=1,3=0"` | Weight by source image count. `0=` T2I, `1=` single-source, `2+` multi-source. |

## Monitoring

| Argument | Description |
|----------|-------------|
| `--report_to` | Logger: `wandb` (default) or `tensorboard` |
| `--report_model_info` | Log gradient norms and parameter statistics |

Logs are saved to `<output_dir>/logs/`.

## VRAM Estimation

| Configuration | Estimated VRAM per GPU |
|--------------|----------------------|
| Rank 32, batch 1, 512px, grad checkpoint | ~24 GB |
| Rank 128, batch 1, 512px, grad checkpoint | ~30 GB |
| Rank 128, batch 2, 512px, grad checkpoint | ~36 GB |
| Rank 128, batch 2, 512px, no grad checkpoint | ~45+ GB |

### VRAM Reduction Strategies (Priority Order)

1. Enable `--gradient_checkpointing` (most impactful)
2. Reduce `--train_batch_size` to 1 and increase `--gradient_accumulation_steps`
3. Lower `--lora_r` (32 instead of 128)
4. Reduce `--image_sample_size` from 512 to 384
5. Lower `--vae_mini_batch` value

## GPU Recommendations

| GPU | VRAM | Feasibility |
|-----|------|-------------|
| RTX 3090 / 4090 | 24 GB | Tight — rank 32, batch 1, grad checkpoint |
| A100 40 GB | 40 GB | Comfortable — rank 128, batch 1-2 |
| A100 80 GB | 80 GB | Full official settings |
| H100 80 GB | 80 GB | Best performance |

### Single GPU Adjustments

When training on a single GPU instead of the official 8-GPU setup:
- Set `--nproc_per_node=1`
- Lower `--train_batch_size` to `1`
- Increase `--gradient_accumulation_steps` to `8` (to match effective batch of 16)
- Reduce `--image_sample_size` to 384 if VRAM is tight

## Complete Recommended Configuration

### Minimal (Single 24 GB GPU)

```
--use_peft_lora
--lora_r 32 --lora_alpha 32 --lora_dropout 0.0
--train_batch_size 1
--gradient_accumulation_steps 8
--image_sample_size 384
--gradient_checkpointing
--mixed_precision bf16
--learning_rate 2e-05
--lr_scheduler constant_with_warmup
--lr_warmup_steps 100
--max_grad_norm 0.05
--adam_weight_decay 3e-2
--adam_epsilon 1e-10
```

### Standard (Single 40+ GB GPU)

```
--use_peft_lora
--lora_r 128 --lora_alpha 128 --lora_dropout 0.0
--train_batch_size 2
--gradient_accumulation_steps 4
--image_sample_size 512
--gradient_checkpointing
--mixed_precision bf16
--learning_rate 2e-05
--lr_scheduler constant_with_warmup
--lr_warmup_steps 100
--max_grad_norm 0.05
--adam_weight_decay 3e-2
--adam_epsilon 1e-10
```

### Official (8 GPUs)

```
--use_peft_lora
--lora_r 128 --lora_alpha 128 --lora_dropout 0.0
--train_batch_size 2
--gradient_accumulation_steps 1
--image_sample_size 512
--gradient_checkpointing
--mixed_precision bf16
--learning_rate 2e-05
--lr_scheduler constant_with_warmup
--lr_warmup_steps 100
--max_grad_norm 0.05
--adam_weight_decay 3e-2
--adam_epsilon 1e-10
--condition_encoder_mode sync
```
