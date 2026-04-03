# LoRA Fundamentals

## What is LoRA?

**LoRA** (Low-Rank Adaptation) is a technique for efficiently fine-tuning a pre-trained model. Instead of modifying all the model's parameters (which requires enormous compute and memory), LoRA adds small, trainable "adapter" matrices alongside the frozen original weights.

### Why LoRA for FireRed-Image-Edit?

FireRed-Image-Edit-1.1 is a large model with a diffusion transformer backbone and a Qwen2.5-VL 7B encoder. Full fine-tuning would require massive GPU resources. LoRA lets you:

- **Train on fewer GPUs** — single GPU is feasible with appropriate settings
- **Save small adapter files** — only the LoRA weights are saved, not the full model
- **Preserve base model capabilities** — the original editing abilities remain intact
- **Iterate quickly** — train multiple adapters for different editing styles

### How It Works

1. You prepare source/target image pairs with editing instructions
2. The Qwen2-VL encoder processes these into embeddings (Stage 1, done offline)
3. The diffusion transformer trains LoRA adapter weights using these embeddings (Stage 2)
4. Only the small adapter weights are saved (~50-500 MB depending on rank)
5. At inference, the adapter weights are loaded alongside the base model

### The Rank Parameter

LoRA works by decomposing weight updates into two smaller matrices. The **rank** (`r`) controls their size:

| Rank | Parameters | Capacity | Training Speed |
|------|-----------|----------|---------------|
| 8 | Fewest | Low — simple adaptations only | Fastest |
| 32 | Low-Medium | Moderate — light style changes | Fast |
| 64 | Medium | Good — most adaptations | Moderate |
| 128 | High | Maximum — new editing domains | Slower |

FireRed's official example uses rank 128 for maximum capacity. DiffSynth-Studio defaults to 32 for lighter adaptations.

### The Alpha Parameter

Alpha (`lora_alpha`) is a scaling factor. The effective LoRA scaling is `alpha / rank`:

| Setting | Effective Scale | Effect |
|---------|----------------|--------|
| alpha = rank | 1.0 | Standard — the official FireRed default |
| alpha = rank / 2 | 0.5 | Conservative — gentler adaptation |
| alpha = rank * 2 | 2.0 | Aggressive — stronger adaptation |

**FireRed default:** `alpha = rank` (both set to 128 or both set to 32).

### Overfitting vs Underfitting

| Symptom | Problem | Fix |
|---------|---------|-----|
| Output copies training edits exactly | Overfitting | Fewer steps, lower LR, more data |
| Instruction has no effect on output | Underfitting | More steps, higher LR, check embeddings |
| Works but only partially | Good starting point | Continue training or adjust parameters |
| Artifacts / distortion | Overfitting or data quality | Earlier checkpoint, audit dataset |

### LoRA vs Full Fine-Tuning for FireRed

| Aspect | LoRA | Full Fine-Tune |
|--------|------|----------------|
| VRAM per GPU | 24-40 GB | 40-80+ GB |
| Multi-GPU | DDP (any GPU count) | FSDP (8+ GPUs recommended) |
| Training time | Hours | Days |
| Output size | 50-500 MB adapter | Full model copy |
| Base model preserved | Yes | No |
| Official support | `train_lora.sh` | `train.sh` |
