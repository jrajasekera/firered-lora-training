# Dataset Format

## JSONL Structure

The dataset must be in JSONL format — one JSON object per line. Each line represents an image editing example.

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_image` | `str`, `list[str]`, or `null` | Yes | Path(s) to source image(s). `null` for text-to-image. |
| `target_image` | `str` | Yes | Path to the ground-truth edited output image. |
| `instruction` | `str` | Yes | English instruction describing the edit. |
| `instruction_cn` | `str` | Recommended | Chinese translation of the instruction. |
| `inverse_instruction` | `str` | Recommended | English description of the *target* image as a standalone caption. |
| `inverse_instruction_cn` | `str` | Recommended | Chinese translation of the inverse instruction. |

Field names are configurable via CLI arguments, but the defaults above are used by the official scripts.

### Example: Standard Image Editing

```jsonl
{"source_image": "/data/img/001.png", "target_image": "/data/img/001_edit.png", "instruction": "Change the sky to sunset.", "instruction_cn": "\u628a\u5929\u7a7a\u6539\u6210\u65e5\u843d\u3002", "inverse_instruction": "A photo of a landscape with a blue sky.", "inverse_instruction_cn": "\u4e00\u5f20\u84dd\u5929\u4e0b\u7684\u98ce\u666f\u7167\u3002"}
```

### Example: Multi-Source Image Editing

```jsonl
{"source_image": ["/data/ref1.png", "/data/ref2.png"], "target_image": "/data/out.png", "instruction": "Merge the two characters into one scene.", "instruction_cn": "\u628a\u4e24\u4e2a\u89d2\u8272\u5408\u6210\u5230\u4e00\u4e2a\u573a\u666f\u91cc\u3002"}
```

### Example: Text-to-Image (No Source)

```jsonl
{"source_image": null, "target_image": "/data/generated.png", "instruction": "A cat sitting on a windowsill.", "instruction_cn": "\u4e00\u53ea\u732b\u5750\u5728\u7a97\u53f0\u4e0a\u3002"}
```

## Writing Good Instructions

### Do

- Be **concise and directive**: *"Change the jacket to red leather with silver zippers"*
- Be **specific about the edit**: *"Convert to watercolor style with soft edges"*
- Describe the **change**, not the final state
- Provide **both English and Chinese** fields — the VLM dialogue structure benefits from both slots being filled, even if your data is originally monolingual

### Don't

- Use vague instructions: *"Make it better"* or *"Improve the image"*
- Include unrelated instructions that don't match the source-to-target difference
- Leave instruction fields empty — every example needs at minimum the `instruction` field

### The Inverse Instruction

The `inverse_instruction` describes the **target image** as a standalone caption (not the edit). This adds a reverse conditioning signal that helps training stability.

- Example instruction: *"Change the sky to sunset"*
- Example inverse_instruction: *"A photo of a landscape with a dramatic orange sunset sky"*

If you cannot generate inverse instructions, you can disable them with `--disable_inverse` during embedding extraction. However, including them is recommended.

## Directory Layout

### Single Dataset

```
/path/to/your_meta_dir/
  my_dataset/
    output_000.jsonl        # Output from embedding extraction (Step 1)
    embeddings_000/         # Saved .pt embedding files
```

### Multi-Dataset Training

Each dataset is placed in its own subdirectory. The subdirectory name becomes the "task" identifier used in `--train_data_weights`:

```
/path/to/your_meta_dir/
  style_transfer/
    output_000.jsonl
    embeddings_000/
  color_editing/
    output_000.jsonl
    embeddings_000/
  object_removal/
    output_000.jsonl
    embeddings_000/
```

Training command references these by name:
```
--train_data_weights="style_transfer=0.5,color_editing=1.0,object_removal=1.2"
```

### Data Weighting

| Argument | Format | Description |
|----------|--------|-------------|
| `--train_data_weights` | `"taskA=0.5,taskB=1.2"` | Sampling weight per subdirectory. Tasks not listed are excluded. |
| `--train_src_img_num_weights` | `"0=1,1=1,2=1,3=0"` | Weight by number of source images. `0=` is T2I, `1=` is single-source, `2=`/`3=` are multi-source. |

Higher weights mean more frequent sampling. Use weights to balance datasets of different sizes or importance.

## Image Quality Guidelines

- Use **high-resolution** source and target images where the edit is clearly attributable to the instruction
- Low-quality pairs confuse the model about what the instruction means
- Ensure source and target images are **aligned** (same framing, same dimensions) unless the edit intentionally changes composition
- Avoid heavily compressed JPEGs as source/target — use PNG when possible
- Training resolution is controlled by `--image_sample_size` (default 512) — images are resized to this during training

## Dataset Size Considerations

There are no official minimum/maximum dataset sizes published, but based on the official LoRA example (`max_train_steps=512`, `train_batch_size=2`):

| Dataset Size | Training Steps | Notes |
|-------------|---------------|-------|
| 50-200 pairs | 256-512 steps | Good for focused style adaptations |
| 200-1000 pairs | 512-2000 steps | Standard training |
| 1000+ pairs | 2000+ steps | Large-scale adaptation |

The official LoRA Zoo models (CoverCraft, Makeup, Lightning) provide reference points for production-quality datasets and training duration.
