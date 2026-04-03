# Step 1: VLM Embedding Extraction

## Why Extract Embeddings Offline?

FireRed-Image-Edit-1.1 uses Qwen2.5-VL 7B as its vision-language encoder. Running this encoder on every training step would be:

- **Expensive** — 7B parameter forward pass per sample per step
- **Redundant** — the encoder is frozen during LoRA training, so its output never changes
- **Memory-hungry** — loading the VLM alongside the DiT during training would require significantly more VRAM

By pre-computing and caching embeddings to disk, you:
- Remove the VLM from GPU memory during training
- Maximize training throughput
- Enable training on smaller GPUs

## The Extraction Script

The script `src/extract_vlm_embeds.py` is run via `torchrun` from the `train/` directory.

### Single-GPU Command

```bash
cd FireRed-Image-Edit/train/

torchrun --nproc_per_node=1 -m src.extract_vlm_embeds \
  /path/to/your/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
```

### Multi-GPU Command (Single Node)

```bash
cd FireRed-Image-Edit/train/

torchrun --nproc_per_node=8 --nnodes 1 --node_rank 0 \
  --master_port 6003 --master_addr localhost \
  -m src.extract_vlm_embeds \
  /path/to/your/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
```

### Multi-Node Command

```bash
cd FireRed-Image-Edit/train/

torchrun --nproc_per_node=8 --nnodes $WORLD_SIZE --node_rank $RANK \
  --master_port $MASTER_PORT --master_addr $MASTER_ADDR \
  -m src.extract_vlm_embeds \
  /path/to/your/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
```

Set the distributed environment variables before launching:
```bash
export WORLD_SIZE=<total_nodes>
export RANK=<current_node_rank>
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=6003
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `jsonl_path` | Required (positional) | Path to your input JSONL file |
| `--output_jsonl_dir` | Required | Directory to write per-rank output JSONL files |
| `--embeddings_save_dir` | Required | Directory to save `.pt` embedding tensors |
| `--model_path` | `/dev/shm/FireRed-Image-Edit-1.0` | Path to FireRed-Image-Edit-1.0 model |
| `--batch_size` | `4` | Batch size for VLM encoding |
| `--image_sample_size` | `512` | Resize images to this square size before encoding |
| `--disable_inverse` | flag | Skip encoding inverse instructions (reduces disk usage) |
| `--t2i_mode` | flag | Text-to-image mode (no source image). Implies `--disable_inverse` |

## What Gets Saved

For each sample, the script saves:

1. **A `.pt` file** containing the Qwen2-VL multimodal embedding tensors
2. **Updated JSONL files** (one per rank) recording paths to the saved embeddings

### Output Structure

```
/path/to/output_jsonl/
  output_000.jsonl      # Rank 0 output
  output_001.jsonl      # Rank 1 output (if multi-GPU)
  ...

/path/to/embeddings/
  embedding_000000.pt
  embedding_000001.pt
  ...
```

## Preparing for Training

After extraction, organize the output into the meta directory structure expected by the training script:

```
/path/to/your_meta_dir/
  my_dataset/
    output_000.jsonl        # Copy from output_jsonl_dir
    embeddings_000/         # Copy or symlink from embeddings_save_dir
      embedding_000000.pt
      embedding_000001.pt
      ...
```

If you used multi-GPU extraction, you'll have multiple output JSONL files (`output_000.jsonl`, `output_001.jsonl`, etc.). All of them should be placed in the dataset subdirectory.

## Important Notes

- **Run from the `train/` directory** so that `src` resolves correctly as a Python package
- **The 1.0 model is required** for extraction (it contains the Qwen2VLProcessor)
- **Disk space:** Each `.pt` embedding file is typically a few MB. Plan for ~2-10 GB for a large dataset.
- **Extraction is idempotent:** Re-running on the same data will overwrite existing outputs
- **Batch size:** Higher batch sizes speed up extraction but use more VRAM. Start with 4 and adjust.

## Condition Encoder Mode

The `--condition_encoder_mode` argument in the training script controls how embeddings are used:

| Mode | Description | When to Use |
|------|-------------|-------------|
| `"offline"` | Use only pre-extracted embeddings | Fastest, requires complete extraction |
| `"sync"` | Run VLM encoder in-process during training | **Used in official LoRA script** |
| `"online"` | Asynchronous online encoding | Not recommended for LoRA |

The official LoRA training script uses `"sync"` mode. Even with `"sync"`, running Stage 1 first is recommended as a validation step to catch data issues early.
