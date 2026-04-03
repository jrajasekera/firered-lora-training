# FireRed-Image-Edit-1.1 LoRA Training Skill

An agent skill providing comprehensive guidance for training LoRA adapters on **FireRedTeam/FireRed-Image-Edit-1.1**, a diffusion transformer-based image editing model that uses Qwen2.5-VL 7B as its multimodal vision-language encoder.

## What This Skill Covers

- The two-stage FireRed training pipeline (VLM embedding extraction + LoRA training)
- JSONL dataset format for image editing pairs
- FireRed-specific LoRA parameters and target modules
- VRAM estimation and hardware planning
- Inference with trained adapters (PEFT + DiffSynth-Studio)
- Alternative training via DiffSynth-Studio
- General LoRA fundamentals and evaluation methodology
- Failure diagnosis and troubleshooting

## Installation

### Hermes Agent

```bash
hermes skills install jrajasekera/firered-lora-training/skills/firered-lora-training
```

### Manual

Copy the `skills/firered-lora-training/` directory into your agent's skills directory:

```
your-project/
  .claude/skills/firered-lora-training/
    SKILL.md
    skill.yaml
    references/
      ...
```

## Structure

```
skills/firered-lora-training/
  SKILL.md                        # Main skill instructions
  skill.yaml                      # Metadata
  references/
    lora-fundamentals.md          # What LoRA is, how it works
    dataset-format.md             # JSONL format, instructions, directory layout
    embedding-extraction.md       # Step 1: VLM embedding extraction
    training-parameters.md        # All training args with FireRed defaults
    evaluation.md                 # Evaluating trained LoRAs
    inference.md                  # Loading and using trained adapters
    alternative-diffsynth.md      # DiffSynth-Studio training path
    failure-modes.md              # Diagnosis and fixes
    troubleshooting.md            # Environment, CUDA, common errors
```

## Key Differences from Generic LoRA Training

FireRed-Image-Edit-1.1 has a unique training pipeline:

1. **Two-stage workflow** — VLM embeddings are extracted offline before training begins
2. **DiT architecture** — uses diffusion transformer blocks, not UNet, with specific target modules
3. **Aggressive gradient clipping** — `max_grad_norm=0.05` (much lower than typical 1.0)
4. **bf16 required** — fp16 is not supported by this architecture
5. **DDP only for LoRA** — FSDP is not recommended for LoRA training on this model
6. **Bilingual conditioning** — both English and Chinese instruction fields strengthen training

## License

MIT
