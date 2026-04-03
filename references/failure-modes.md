# Failure Modes: Diagnosis and Fixes

## Quick Diagnosis Table

| Symptom | Likely Cause | Priority Fix |
|---------|-------------|-------------|
| NaN loss immediately | Learning rate too high | Lower LR to 1e-5, check `max_grad_norm` |
| Loss not decreasing | Adapter not applied, LR too low | Verify `--use_peft_lora`, check trainable params |
| Edits copy training data | Overfitting | Earlier checkpoint, reduce steps |
| Edits are too weak | Underfitting | More steps, check embeddings exist |
| Wrong edits applied | Dataset quality issue | Audit instruction-edit alignment |
| Artifacts in output | Overfitting or bad data | Earlier checkpoint, audit images |
| OOM during training | VRAM insufficient | Reduce batch, enable grad checkpoint |
| OOM during extraction | VLM batch too large | Reduce `--batch_size` in extraction |

---

## Detailed Analysis

### NaN Loss

**What's happening:** Training has diverged — weights have become nonsensical.

**Causes specific to FireRed:**
- Learning rate too high for this architecture (even 1e-4 can be too high with official settings)
- `max_grad_norm` set higher than 0.05
- Corrupted images in dataset causing invalid tensors
- Missing or corrupted embedding files

**Fixes (priority order):**
1. Verify `--max_grad_norm=0.05` (the official value)
2. Lower `--learning_rate` to `1e-5`
3. Check for corrupted images in dataset
4. Re-run embedding extraction for problematic samples
5. Verify `--mixed_precision=bf16` (not fp16)

### Loss Not Decreasing

**What's happening:** The model isn't learning — loss stays flat.

**Causes:**
- `--use_peft_lora` flag missing (training in full fine-tune mode with wrong settings)
- Learning rate too low
- Adapter not targeting correct modules
- Embeddings not loading correctly

**Fixes (priority order):**
1. Verify `--use_peft_lora` is present in command
2. Check training logs for trainable parameter count (should be > 0)
3. Increase `--learning_rate` to `5e-5`
4. Verify embedding files exist and are readable
5. Check `--train_data_weights` includes your dataset

### Edits Copy Training Data Exactly

**What's happening:** The model has memorized training pairs instead of learning the editing concept.

**Causes:** Overfitting — too many steps, too small dataset, or rank too high.

**Fixes (priority order):**
1. Use an earlier checkpoint (try 50% of total steps)
2. Reduce `--max_train_steps`
3. Add more diverse training pairs
4. Lower rank from 128 to 64 or 32

### Edits Are Too Weak / Instruction Ignored

**What's happening:** The LoRA has minimal effect — output looks similar to base model.

**Causes:** Underfitting, embedding issues, or low adapter weight during inference.

**Fixes (priority order):**
1. Increase `--max_train_steps` by 50-100%
2. Verify embeddings were extracted correctly (check `.pt` files exist)
3. Ensure `--condition_encoder_mode` is set to `"sync"` or `"offline"` (not missing)
4. Try higher LoRA weight during inference (1.0)
5. Increase rank if using 32 — try 64 or 128

### Wrong Edits Being Applied

**What's happening:** The model edits the image, but not in the way the instruction describes.

**Causes:** Dataset quality — instructions don't match the actual source-to-target difference.

**Fixes (priority order):**
1. Audit dataset: for each pair, verify the instruction accurately describes the change
2. Remove ambiguous pairs where the edit isn't clearly attributable to the instruction
3. Add `inverse_instruction` fields if not present (provides additional conditioning signal)
4. Ensure image pairs are properly aligned (same framing/dimensions)

### Artifacts / Distortion

**What's happening:** Output has visual glitches — color banding, noise, warping in unedited regions.

**Causes:** Overfitting, bad training data, or incorrect precision.

**How to check:**
- Artifacts at all checkpoints → data quality issue
- Artifacts only at late checkpoints → overfitting

**Fixes (priority order):**
1. Use an earlier checkpoint
2. Audit images for JPEG compression artifacts, low resolution, or corruption
3. Verify `--mixed_precision=bf16` (fp16 can cause numerical issues)
4. Reduce training steps
5. Lower rank

### Partial Generalization

**What's happening:** LoRA works for some instructions but fails for others.

**Causes:** Dataset doesn't cover enough diversity of editing types.

**Fixes:**
1. Add more diverse training pairs covering different edit types
2. Balance dataset — don't let one edit type dominate
3. Increase training steps slightly (the model may need more exposure)

## FireRed-Specific Gotchas

### fp16 vs bf16

This architecture **requires bf16**. Using fp16 will cause:
- NaN loss
- Incorrect gradients
- Garbage output

Always verify `--mixed_precision="bf16"` in your training command.

### FSDP with LoRA

The codebase warns: *"LoRA training is not recommended with FSDP."*

If you accidentally enable `--use_fsdp`, LoRA training may:
- Fail to save checkpoints correctly
- Produce degraded results
- Have parameter synchronization issues

Use DDP (the default for LoRA) instead.

### Condition Encoder Mode

| Mode | Issue If Wrong |
|------|---------------|
| `"offline"` without Step 1 done | Training crashes — embeddings not found |
| `"online"` for LoRA | Not recommended, may cause instability |
| `"sync"` (recommended) | Most reliable for LoRA training |

### Gradient Clipping Too High

If you raise `--max_grad_norm` above 0.05:
- Early training steps may see gradient explosions
- Loss may spike or go to NaN
- Modulation layers are particularly sensitive

The aggressive 0.05 default is intentional for this architecture's gradient dynamics.
