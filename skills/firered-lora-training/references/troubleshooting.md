# Troubleshooting

## Environment Issues

### Python Version

FireRed-Image-Edit-1.1 training requires **Python 3.12**.

```bash
python --version
# Must be 3.12.x
```

### flash-attn Build Errors

**Error:** Build fails during `pip install flash-attn`

Flash Attention requires a compatible CUDA environment. If your CUDA version differs from `cu128`:

```bash
# Check your CUDA version
nvidia-smi
nvcc --version

# Build from source with your CUDA version
pip install flash-attn --no-build-isolation
```

If build still fails, ensure you have the CUDA toolkit development headers installed.

### Triton Not Available

**Error:** `ModuleNotFoundError: No module named 'triton'`

Triton is Linux-only:
```bash
pip install triton  # Linux only
```

On macOS/Windows, Triton is not available. Training must be done on Linux.

### torch + CUDA Mismatch

**Error:** `CUDA error: no kernel image is available`

Ensure your torch version matches your CUDA:
```bash
python -c "import torch; print(torch.version.cuda)"
# Should match your nvidia-smi CUDA version
```

The official requirements specify `torch==2.9.0+cu128`. If your CUDA differs:
```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

---

## Model Download Issues

### HuggingFace Authentication

If downloads fail, authenticate first:
```bash
huggingface-cli login
```

### Model Directory Structure

Verify the expected structure after download:

```
FireRed-Image-Edit-1.0/
  transformer/
  vae/
  processor/          # Qwen2VLProcessor lives here

FireRed-Image-Edit-1.1/
  transformer/
    diffusion_pytorch_model.safetensors
```

If `processor/` is missing from the 1.0 model, embedding extraction will fail.

---

## Embedding Extraction Issues

### "Module 'src' not found"

**Cause:** Not running from the `train/` directory.

**Fix:**
```bash
cd FireRed-Image-Edit/train/
torchrun ... -m src.extract_vlm_embeds ...
```

### OOM During Extraction

**Fix:** Reduce `--batch_size`:
```bash
--batch_size 1  # Minimum
```

### Extraction Hangs

**Possible causes:**
- Corrupted image file causing infinite decode loop
- Network file system (NFS) latency
- Distributed setup misconfiguration

**Fix:** Try single-GPU extraction first (`--nproc_per_node=1`) to isolate the issue. Check for corrupted images by loading each with PIL:

```python
from PIL import Image
import json

with open("data.jsonl") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        for img_path in [entry.get("source_image"), entry.get("target_image")]:
            if img_path and isinstance(img_path, str):
                try:
                    Image.open(img_path).verify()
                except Exception as e:
                    print(f"Line {i}: Bad image {img_path}: {e}")
```

### Missing Embedding Files

If training reports missing `.pt` files:
1. Check the output JSONL references valid paths
2. Verify the embedding directory structure matches what training expects
3. Re-run extraction if needed (it's idempotent)

---

## Training Issues

### CUDA OOM During Training

**Fixes (try in order):**
1. Enable `--gradient_checkpointing` (if not already)
2. Reduce `--train_batch_size` to `1`
3. Increase `--gradient_accumulation_steps` to compensate
4. Lower `--lora_r` (32 instead of 128)
5. Reduce `--image_sample_size` from 512 to 384

### "LoRA training is not recommended with FSDP" Warning

This is expected if you accidentally included `--use_fsdp`. Remove the FSDP flags:

```bash
# Remove these from your command:
# --use_fsdp
# --fsdp_auto_wrap_policy
# --fsdp_transformer_layer_cls_to_wrap
# --fsdp_state_dict_type
```

LoRA training uses DDP automatically.

### Checkpoints Not Being Saved

**Check:**
1. `--output_dir` exists and is writable
2. `--checkpointing_steps` is set (default may be very high)
3. Disk space is sufficient
4. Training has actually reached the checkpoint step count

### Training Crashes After Checkpoint

**If resuming fails:**
```bash
# Try resuming from latest
--resume_from_checkpoint "latest"

# Or from a specific checkpoint
--resume_from_checkpoint "/path/to/ckpts_lora/checkpoint-100"
```

Verify the checkpoint contains both `adapter_config.json` and `adapter_model.safetensors`.

### fp16 Errors

**Error:** Various numerical errors, NaN loss, or incorrect gradients

This architecture does **not support fp16**. Ensure:
```bash
--mixed_precision="bf16"
```

Check your GPU supports bf16 (Ampere / A100 / RTX 30xx+ and newer).

---

## Inference Issues

### Adapter Loading Fails

**Error:** `ValueError: Can't find adapter weights`

**Check:**
1. Checkpoint directory contains `adapter_config.json` + `adapter_model.safetensors`
2. Path is correct (absolute paths are safest)
3. You're loading onto the correct base model architecture

### Output Quality Differs from Training

**Common causes:**
- Wrong base model (using 1.0 instead of 1.1 transformer)
- Wrong precision (fp32 or fp16 instead of bf16)
- Inference resolution very different from training resolution
- LoRA weight/strength not set correctly

### DiffSynth vs PEFT Loading Confusion

The two training paths produce different checkpoint formats:

| Pipeline | Format | How to Load |
|----------|--------|-------------|
| Official FireRedTeam | PEFT adapter directory | `PeftModel.from_pretrained()` |
| DiffSynth-Studio | Single `.safetensors` | `load_state_dict()` + `strict=False` |

Don't mix them — use the loading method that matches your training pipeline.

---

## Distributed Training Issues

### NCCL Errors

**Error:** `NCCL error: unhandled system error`

**Fixes:**
```bash
# Set NCCL debug logging
export NCCL_DEBUG=INFO

# Try different NCCL socket interface
export NCCL_SOCKET_IFNAME=eth0

# Disable IB if not available
export NCCL_IB_DISABLE=1
```

### Rank Mismatch Errors

Ensure all nodes have consistent:
- `WORLD_SIZE`
- `MASTER_ADDR` (reachable from all nodes)
- `MASTER_PORT` (not blocked by firewall)
- Same code version and dependencies

---

## Performance Issues

### Training Very Slow

**Check:**
1. TF32 enabled: `--allow_tf32`
2. Flash Attention installed: `python -c "import flash_attn"`
3. `--dataloader_num_workers` > 0 for disk-bound datasets
4. GPU utilization (should be 90%+): `nvidia-smi`

### GPU Utilization Low

**Causes:**
- Data loading is the bottleneck → increase `--dataloader_num_workers`
- Gradient checkpointing overhead → expected, but verify it's not excessive
- Small dataset with high batch → steps finish too quickly for GPU to saturate

---

## Debug Checklist

When reporting issues, include:

```
Environment:
- Python: X.Y.Z
- PyTorch: X.Y.Z+cuXXX
- PEFT: X.Y.Z
- Transformers: X.Y.Z
- Diffusers: X.Y.Z
- GPU: NVIDIA XXXX (XX GB) x N
- CUDA: XX.X

Training Config:
- lora_r: X
- lora_alpha: X
- learning_rate: X
- max_train_steps: X
- train_batch_size: X
- condition_encoder_mode: X
- mixed_precision: bf16

Error:
<paste full traceback>

Training logs:
<paste last 20 lines of log output>
```
