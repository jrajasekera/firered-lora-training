# Inference with Trained LoRAs

## Method 1: Using PEFT Directly (Diffusers)

For adapters trained with the official FireRedTeam pipeline:

```python
from peft import PeftModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
import torch

# Load base transformer
transformer = QwenImageTransformer2DModel.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter
transformer = PeftModel.from_pretrained(
    transformer,
    "/path/to/ckpts_lora/checkpoint-200"
)

# Optional: merge adapter into base weights for faster inference
transformer = transformer.merge_and_unload()
```

### Loading a Specific Checkpoint

```python
# Load checkpoint at step 100
transformer = PeftModel.from_pretrained(transformer, "/path/to/ckpts_lora/checkpoint-100")

# Or latest
transformer = PeftModel.from_pretrained(transformer, "/path/to/ckpts_lora/checkpoint-latest")
```

## Method 2: Using DiffSynth-Studio

For adapters trained via the DiffSynth-Studio path:

```python
import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict

# Build pipeline
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="FireRedTeam/FireRed-Image-Edit-1.1",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="text_encoder/model*.safetensors"
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors"
        ),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Edit",
        origin_file_pattern="processor/"
    ),
)

# Load trained LoRA weights
state_dict = load_state_dict(
    "models/train/FireRed-Image-Edit-1.1_lora/epoch-5.safetensors"
)
pipe.dit.load_state_dict(state_dict, strict=False)

# Run inference
prompt = "Change the jacket color to red."
image = Image.open("my_photo.jpg")
result = pipe(
    prompt,
    edit_image=image,
    seed=42,
    num_inference_steps=40,
    height=1024,
    width=1024,
)
result.save("output.jpg")
```

## Merging for Deployment

Merging bakes the LoRA weights into the base model, eliminating adapter overhead at inference:

```python
from peft import PeftModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
import torch

# Load and merge
transformer = QwenImageTransformer2DModel.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
transformer = PeftModel.from_pretrained(transformer, "/path/to/adapter")
merged = transformer.merge_and_unload()

# Save merged model
merged.save_pretrained("/path/to/merged_transformer")
```

**Trade-offs of merging:**
- Pro: No adapter loading overhead, simpler deployment
- Pro: Faster inference (no LoRA forward pass)
- Con: Can't easily adjust LoRA weight/strength
- Con: Can't swap adapters dynamically
- Con: Larger file size (full model instead of small adapter)

## Resuming Training from a Checkpoint

To continue training from a saved adapter:

```bash
--use_peft_lora \
--lora_r 128 \
--lora_alpha 128 \
--lora_path /path/to/ckpts_lora/checkpoint-200/adapter_model.safetensors
```

The loading logic supports both:
- A **directory** path (looks for `adapter_model.safetensors` then `adapter_model.bin`)
- A **direct file** path to the safetensors/bin file

To resume full training state (optimizer, scheduler, step count):
```bash
--resume_from_checkpoint "latest"
# or
--resume_from_checkpoint "/path/to/ckpts_lora/checkpoint-200"
```

## Checkpoint File Format

FireRed LoRA checkpoints (non-FSDP) are standard PEFT adapter files:

```
checkpoint-200/
  adapter_config.json        # LoRA configuration (rank, alpha, target modules)
  adapter_model.safetensors  # Adapter weights
```

These are compatible with the standard `peft` library — no custom loading code needed.

## Important Notes

- **Always use bf16** for inference, matching the training precision
- **The 1.0 model is still needed** at inference for the VAE and processor, even when using 1.1 transformer weights
- **Image resolution** at inference can differ from training resolution (512) — the model handles different sizes
- **Inference steps:** Default is ~40 steps. The Lightning LoRA from the official LoRA Zoo reduces this to 8 steps via distillation
