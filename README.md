# runpod-serverless-generation

RunPod Serverless worker for image and video generation. Uses pre-downloaded
models from a RunPod Network Volume and stores results on the same volume.

The worker **does not download any models**. All required model files must
already exist under `/runpod-volume/models`.

---

## Features

- Action `health` — liveness check.
- Action `generate` — run a generation workflow.
- Workflows:
  - `flux2_reactor` — FLUX.2 image generation with optional ReActor face swap.
  - `wan22_i2v` — Wan 2.2 image-to-video generation.
- Atomic file writes, metadata JSON, automatic previews.
- Path-traversal-safe LoRA names and unified error format.

---

## Project structure

```text
.
  README.md
  Dockerfile
  requirements.txt

  src/
    handler.py
    config.py
    schemas.py
    storage.py
    model_paths.py

    generators/
      __init__.py
      flux2_reactor.py
      wan22_i2v.py

    utils/
      __init__.py
      paths.py
      atomic.py
      images.py
      video.py
      logging.py
      base64_utils.py
```

---

## Environment variables

| Variable            | Default                       | Description                          |
|---------------------|-------------------------------|--------------------------------------|
| `RUNPOD_VOLUME_DIR` | `/runpod-volume`              | Root of the network volume.          |
| `MODEL_BASE_DIR`    | `/runpod-volume/models`       | Where model files are located.       |
| `STORAGE_BASE_DIR`  | `/runpod-volume/storage`      | Where generation output is written.  |
| `TMP_DIR`           | `/runpod-volume/tmp`          | Temporary files during generation.   |

---

## Expected model layout

```text
/runpod-volume/models/
  diffusion_models/
    flux2_dev_fp8mixed.safetensors
    wan2.2_ti2v_5B_fp16.safetensors
  text_encoders/
    mistral_3_small_flux2_bf16.safetensors
    umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/
    flux2-vae.safetensors
    wan2.2_vae.safetensors
  insightface/
    buffalo_l.zip
    inswapper_128.onnx
  loras/
    <your_lora_files>.safetensors
  checkpoints/
  custom/
```

### Required files per workflow

**`flux2_reactor`**

- `diffusion_models/flux2_dev_fp8mixed.safetensors`
- `text_encoders/mistral_3_small_flux2_bf16.safetensors`
- `vae/flux2-vae.safetensors`
- `insightface/buffalo_l.zip`
- `insightface/inswapper_128.onnx`

**`wan22_i2v`**

- `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
- `vae/wan2.2_vae.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`

If any required file is missing, the worker returns `MODEL_NOT_FOUND`.

---

## Expected storage layout

```text
/runpod-volume/storage/
  inputs/
  images/YYYY/MM/DD/<id>_output.png
  videos/YYYY/MM/DD/<id>_result.mp4
  previews/YYYY/MM/DD/<id>_preview.jpg
  metadata/YYYY/MM/DD/<id>.json
```

All `*_relative_path` fields in responses are relative to `STORAGE_BASE_DIR`.

Temporary files are written under `/runpod-volume/tmp/generation/`.

---

## Downloading models onto the volume

The worker itself never downloads anything. To populate the volume, use the
helper script [`scripts/download_model.py`](scripts/download_model.py). Run
it from a pod (or any machine) that has the network volume mounted.

```bash
pip install -r scripts/requirements.txt

# HuggingFace
python scripts/download_model.py \
  https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/flux2_dev_fp8mixed.safetensors \
  /runpod-volume/models/diffusion_models

# Civitai (token usually required)
CIVITAI_TOKEN=xxx python scripts/download_model.py \
  https://civitai.com/api/download/models/123456 \
  /runpod-volume/models/loras \
  --filename portrait_style_lora.safetensors

# Direct URL
python scripts/download_model.py \
  https://example.com/path/to/model.safetensors \
  /runpod-volume/models/custom
```

The script shows a live progress bar, writes to a `.part` temp file and
atomically renames on success. Tokens can also come from env:
`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` and `CIVITAI_TOKEN`.

---

## Build the Docker image

```bash
docker build -t runpod-serverless-generation:latest .
```

The base image is `python:3.12-slim`. If GPU/CUDA is required for real
generators, swap the base image for a CUDA-compatible one (for example
`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with Python installed).

---

## Deploy on RunPod Serverless

The repository layout is already compatible with the **RunPod → Serverless →
Deploy from GitHub** flow — `Dockerfile` and `requirements.txt` live at the
repo root.

### Option A: Deploy from GitHub

1. Push this repository to GitHub.
2. Open **RunPod → Serverless → New Endpoint → Deploy from GitHub**.
3. Select this repository and branch.
4. RunPod will build the image from the root `Dockerfile` automatically.
5. In the endpoint settings:
   - attach your **Network Volume** (must be mounted at `/runpod-volume`,
     which is the RunPod default),
   - pick a GPU suitable for your workflow,
   - leave the default start command (the `Dockerfile` sets it to
     `python -u -m src.handler`).
6. Make sure the models already exist on the volume under
   `/runpod-volume/models/...` (this worker does **not** download them).

### Option B: Build and push a Docker image manually

```bash
docker build -t <your-registry>/runpod-serverless-generation:latest .
docker push <your-registry>/runpod-serverless-generation:latest
```

Then in RunPod create a Serverless endpoint from that image and attach the
network volume as above.

### Environment variables on RunPod

All variables have sensible defaults matching the RunPod volume layout, so
usually nothing needs to be set. Override only if your paths differ:

```text
RUNPOD_VOLUME_DIR=/runpod-volume
MODEL_BASE_DIR=/runpod-volume/models
STORAGE_BASE_DIR=/runpod-volume/storage
TMP_DIR=/runpod-volume/tmp
```

### Test the endpoint

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"action":"health"}}' \
  https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
```

---

## Run locally

Useful for development and for validating the handler against real request
payloads before deploying.

### 1. Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

`ffmpeg` must also be available on `PATH` for video previews:

- Windows: `winget install Gyan.FFmpeg` (or download from ffmpeg.org).
- macOS: `brew install ffmpeg`.
- Linux: `apt-get install ffmpeg`.

### 2. Prepare a fake volume

The worker expects the same layout locally. Create a folder somewhere on
disk and point env variables at it:

```powershell
# PowerShell
$env:RUNPOD_VOLUME_DIR = "D:\fake-runpod-volume"
$env:MODEL_BASE_DIR    = "$env:RUNPOD_VOLUME_DIR\models"
$env:STORAGE_BASE_DIR  = "$env:RUNPOD_VOLUME_DIR\storage"
$env:TMP_DIR           = "$env:RUNPOD_VOLUME_DIR\tmp"

# Create the directory layout
New-Item -ItemType Directory -Force `
  "$env:MODEL_BASE_DIR\diffusion_models", `
  "$env:MODEL_BASE_DIR\text_encoders", `
  "$env:MODEL_BASE_DIR\vae", `
  "$env:MODEL_BASE_DIR\insightface", `
  "$env:MODEL_BASE_DIR\loras", `
  "$env:STORAGE_BASE_DIR" | Out-Null
```

```bash
# bash
export RUNPOD_VOLUME_DIR=/tmp/fake-runpod-volume
export MODEL_BASE_DIR=$RUNPOD_VOLUME_DIR/models
export STORAGE_BASE_DIR=$RUNPOD_VOLUME_DIR/storage
export TMP_DIR=$RUNPOD_VOLUME_DIR/tmp

mkdir -p $MODEL_BASE_DIR/{diffusion_models,text_encoders,vae,insightface,loras} $STORAGE_BASE_DIR
```

To exercise the `generate` action without real weights, create **empty
placeholder files** with the expected names so `verify_required_models`
passes:

```bash
# flux2_reactor
touch $MODEL_BASE_DIR/diffusion_models/flux2_dev_fp8mixed.safetensors \
      $MODEL_BASE_DIR/text_encoders/mistral_3_small_flux2_bf16.safetensors \
      $MODEL_BASE_DIR/vae/flux2-vae.safetensors \
      $MODEL_BASE_DIR/insightface/buffalo_l.zip \
      $MODEL_BASE_DIR/insightface/inswapper_128.onnx

# wan22_i2v
touch $MODEL_BASE_DIR/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
      $MODEL_BASE_DIR/vae/wan2.2_vae.safetensors \
      $MODEL_BASE_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

The generators are MVP stubs, so empty files are enough to pass the
existence checks.

### 3. Run the handler directly (RunPod local mode)

```bash
python -m src.handler
```

RunPod's SDK provides a local test mode — pass a JSON payload via
`--test_input`:

```bash
python -m src.handler --test_input '{"input":{"action":"health"}}'
```

```bash
python -m src.handler --test_input '{
  "input": {
    "action": "generate",
    "workflow_type": "flux2_reactor",
    "prompt": "a cat astronaut"
  }
}'
```

### 4. Or call the handler function directly

```python
from src.handler import handler

print(handler({"input": {"action": "health"}}))

print(handler({
    "input": {
        "action": "generate",
        "workflow_type": "flux2_reactor",
        "prompt": "a cat astronaut",
    }
}))
```

### 5. Run inside Docker locally

```bash
docker build -t runpod-serverless-generation:latest .

docker run --rm -it \
  -v /tmp/fake-runpod-volume:/runpod-volume \
  runpod-serverless-generation:latest \
  python -m src.handler --test_input '{"input":{"action":"health"}}'
```

---


## RunPod job examples

### Health

```json
{
  "input": {
    "action": "health"
  }
}
```

Response:

```json
{
  "ok": true,
  "service": "generation-worker"
}
```

### FLUX.2 image

```json
{
  "input": {
    "action": "generate",
    "workflow_type": "flux2_reactor",
    "prompt": "realistic portrait photo, soft studio light",
    "negative_prompt": "bad quality, blurry",
    "seed": 12345,
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance_scale": 3.5,
    "face_image_base64": "data:image/png;base64,...",
    "loras": [
      {
        "name": "portrait_style_lora.safetensors",
        "strength_model": 0.8,
        "strength_clip": 0.8
      }
    ],
    "generation_params": { "sampler": "default" }
  }
}
```

### Wan 2.2 video

```json
{
  "input": {
    "action": "generate",
    "workflow_type": "wan22_i2v",
    "prompt": "cinematic camera movement, realistic motion",
    "negative_prompt": "flickering, distorted",
    "seed": 12345,
    "width": 832,
    "height": 480,
    "frames": 81,
    "fps": 16,
    "input_image_base64": "data:image/png;base64,...",
    "generation_params": { "model": "wan22_ti2v_5b" }
  }
}
```

---

## Success response

Image:

```json
{
  "ok": true,
  "id": "7d2f1c8a9b",
  "type": "image",
  "workflow_type": "flux2_reactor",
  "filename": "7d2f1c8a9b_output.png",
  "relative_path": "images/2026/04/24/7d2f1c8a9b_output.png",
  "preview_relative_path": "previews/2026/04/24/7d2f1c8a9b_preview.jpg",
  "metadata_relative_path": "metadata/2026/04/24/7d2f1c8a9b.json"
}
```

Video:

```json
{
  "ok": true,
  "id": "a91bcf2012",
  "type": "video",
  "workflow_type": "wan22_i2v",
  "filename": "a91bcf2012_result.mp4",
  "relative_path": "videos/2026/04/24/a91bcf2012_result.mp4",
  "preview_relative_path": "previews/2026/04/24/a91bcf2012_preview.jpg",
  "metadata_relative_path": "metadata/2026/04/24/a91bcf2012.json"
}
```

---

## Error response

```json
{
  "ok": false,
  "error": {
    "code": "GENERATION_FAILED",
    "message": "Human readable error",
    "details": {}
  }
}
```

Error codes:

```text
INVALID_ACTION
INVALID_REQUEST
UNSUPPORTED_WORKFLOW
MODEL_NOT_FOUND
LORA_NOT_FOUND
INPUT_IMAGE_REQUIRED
FACE_IMAGE_INVALID
GENERATION_FAILED
SAVE_FAILED
PREVIEW_FAILED
UNKNOWN_ERROR
```

---

## LoRA

LoRA files must be located in `/runpod-volume/models/loras/`.

In a request, the caller passes only the **file name**:

```json
"loras": [
  {
    "name": "portrait_style_lora.safetensors",
    "strength_model": 0.8,
    "strength_clip": 0.8
  }
]
```

Rules:

- absolute paths are rejected,
- `..` is rejected,
- `/` and `\` are rejected,
- subdirectories are rejected.

The final path is always `/runpod-volume/models/loras/<name>`.
If the LoRA file does not exist, the worker returns `LORA_NOT_FOUND`.

---

## Notes

- The worker **does not download models**.
- There is **no File API** and **no Civitai/Hugging Face download logic**.
- Generators in `src/generators/` are currently MVP stubs that produce a
  placeholder PNG / short placeholder MP4. They can be replaced with real
  FLUX.2 + ReActor and Wan 2.2 implementations without touching the handler,
  storage or model-verification layers.
