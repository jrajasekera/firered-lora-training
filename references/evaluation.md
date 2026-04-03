# Evaluating Trained LoRAs

## Evaluation Process

1. **Test across checkpoints** — compare early, mid, and late checkpoints
2. **Test across LoRA weights** — generate at different adapter strengths
3. **Test with diverse prompts** — verify generalization beyond training instructions
4. **Compare to base model** — ensure the LoRA adds value without degrading quality

## Checkpoint Comparison

Checkpoints are saved at intervals defined by `--checkpointing_steps`. The best checkpoint is rarely the last one.

| Checkpoint Timing | Typical Behavior |
|-------------------|-----------------|
| Early (25% of training) | Underfitted — edits are weak |
| Mid (50% of training) | Often the sweet spot |
| Late (75-100% of training) | Risk of overfitting — edits may be rigid |

**Always test multiple checkpoints** before selecting one for production.

### Checkpoint Structure

```
/path/to/ckpts_lora/
  checkpoint-100/
    adapter_config.json
    adapter_model.safetensors
  checkpoint-200/
    adapter_config.json
    adapter_model.safetensors
  checkpoint-latest -> checkpoint-200/
```

## LoRA Weight / Strength

When loading the adapter for inference, you can control its influence. For FireRed LoRAs:

| Weight | Expected Effect |
|--------|----------------|
| 0.3-0.5 | Subtle influence — gentle style shift |
| 0.5-0.7 | Moderate — noticeable but balanced |
| 0.7-0.9 | Strong — clear adaptation |
| 1.0 | Full strength — as trained |

Generate the same editing prompt at each weight to see the progression and find the sweet spot.

## What to Look For

### Edit Fidelity
- Does the LoRA perform the intended type of edit?
- Is the edit precise (changes only what the instruction asks)?
- Does it preserve parts of the image that shouldn't change?

### Quality Preservation
- Any artifacts or distortion in unedited regions?
- Color banding or noise introduced?
- Resolution/sharpness maintained?
- Faces/text/fine details preserved?

### Generalization
- Does it work with instructions not seen during training?
- Does it handle different image types (photos, illustrations, etc.)?
- Does it work at different image resolutions?
- Does it handle edge cases (small edits, large edits)?

## Test Prompt Strategy

Design test prompts that cover:

### 1. In-Distribution Edits
Instructions similar to your training data — these should work well.

### 2. Near-Distribution Edits
Variations on training instructions — tests generalization.

### 3. Out-of-Distribution Edits
Completely new instruction types — tests how well the LoRA maintains base model capabilities.

### Example Test Set for a Style LoRA

```
1. "Convert to watercolor style"               (in-distribution if trained on style)
2. "Apply a painterly effect with soft edges"   (near-distribution)
3. "Change the background to a forest"          (out-of-distribution edit type)
4. "Remove the person from the image"           (base model capability check)
```

## Decision Framework

| Verdict | Symptoms | Action |
|---------|----------|--------|
| **Ready** | Edits are accurate, generalizes to new instructions, no artifacts | Use this checkpoint |
| **Needs tuning** | Partially works, some instructions fail | Try different checkpoint, adjust parameters |
| **Overtrained** | Edits are rigid, artifacts appear, ignores novel instructions | Use earlier checkpoint, reduce training steps |
| **Undertrained** | Edits are weak or inconsistent | Continue training, increase steps |
| **Dataset issue** | Wrong edits learned, inconsistent behavior | Revisit dataset quality and instructions |

## Reference: Official LoRA Zoo

FireRedTeam's published LoRAs serve as quality benchmarks:

| LoRA | Purpose | What to Learn From It |
|------|---------|----------------------|
| **CoverCraft** | Artistic typography, poster fonts | Complex style domain adaptation |
| **Makeup** | Cosmetic editing, skin refinement | Precise localized edits |
| **Lightning** | 8-step distillation | Speed-quality tradeoff |

These are available at [`FireRedTeam/FireRed-Image-Edit-LoRA-Zoo`](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-LoRA-Zoo). Inspect their adapter configs to understand production rank/alpha settings.

## Monitoring During Training

Track these metrics to predict training outcome before evaluation:

| Metric | Healthy | Warning |
|--------|---------|---------|
| Training loss | Decreasing, then plateaus | NaN, increasing, or stuck at initial value |
| Gradient norms | Stable, within clipping threshold | Frequently hitting clip limit, spiking |
| Learning rate | Following scheduler curve | Flat when should be changing |

Enable `--report_model_info` to log gradient norms and parameter statistics for deeper diagnostics.
