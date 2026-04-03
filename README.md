# VoxHub

**Multi-model transcription platform with an OpenAI-compatible API.** Run Voxtral, Whisper, Moonshine, and Canary side-by-side — with advanced VAD, speaker diarization, and optional benchmarking. Optimized for NVIDIA Blackwell (GB10/GX10), CUDA, ROCm, and CPU.

---

## What VoxHub Does

VoxHub is an API-first transcription platform that lets you swap ASR backends without changing your client code. It exposes an OpenAI-compatible `/v1/audio/transcriptions` endpoint, so any client that works with the OpenAI Audio API (OpenHiNotes, TypingMind, etc.) works with VoxHub out of the box.

Under the hood, VoxHub orchestrates multiple transcription engines through a unified pipeline: audio loading, VAD segmentation, model inference, segment merging, and multi-format output — all configurable per request.

---

## Quick Start

### Docker (Recommended)

```bash
# Build the image
docker compose build voxhub-api

# Start the API server on port 8000
docker compose up voxhub-api
```

### Manual

```bash
pip install -r requirements.txt
# Create a .env file with your HF_TOKEN (see .env.example)
python server.py
```

### Test It

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio/meeting.mp3 \
  -F model=whisper:turbo \
  -F response_format=verbose_json
```

---

## Supported Models

| Family | Specifier | Backend | Key Strength |
| :--- | :--- | :--- | :--- |
| **Voxtral** | `voxtral:mini-4b`, `voxtral:small-24b` | Transformers | Real-time native support, HQ French/English |
| **Whisper** | `whisper:large-v3`, `whisper:turbo`, `whisper:small` | Transformers | Global benchmark leader, extreme throughput |
| **Moonshine** | `moonshine:base`, `moonshine:tiny` | Transformers | Ultra-low latency, CPU-efficient |
| **Canary** | `canary:1b` | NeMo | SOTA accuracy, multi-task (ASR/translation) |

Models are defined in `models.yaml` and loaded lazily on first request.

---

## VAD & Diarization

VoxHub supports four VAD strategies, selectable per request or via config:

| Mode | Description | Best For |
| :--- | :--- | :--- |
| `silero` | Fast, lightweight Silero VAD | Speed-first workflows |
| `pyannote` | High-accuracy Pyannote segmentation + speaker diarization | Production quality |
| `hybrid` | Silero gate + Pyannote refiner with confidence-based safety net | Best recall + precision |
| `none` | No segmentation — process entire audio as one segment | Pre-segmented audio |

### Hybrid Mode (Refined Gating)

The hybrid pipeline uses Silero as a sensitive gate (threshold `0.35`) to capture all potential speech, then passes the audio to Pyannote for fine-grained segmentation and speaker labeling. A safety-net override keeps high-confidence Silero segments (probability > `0.8`) even if Pyannote disagrees.

```bash
# CLI
python main.py audio/meeting.mp3 --vad hybrid --diarize

# Tune the thresholds
python main.py audio/meeting.mp3 --vad hybrid --silero-threshold 0.3 --override-threshold 0.85
```

---

## API Reference

### Transcription (Synchronous)

`POST /v1/audio/transcriptions` — OpenAI-compatible. Supports `json`, `verbose_json`, `text`, `srt`, `vtt` formats.

### Transcription (Async Jobs)

For long audio files, use the Jobs API:

1. `POST /v1/audio/transcriptions/jobs` — Submit a job. Returns `job_id` and status/result links.
2. `GET /v1/audio/transcriptions/jobs/{job_id}` — Poll status. Returns `status` (`pending`, `processing`, `completed`, `failed`), `stage` (`loading`, `vad`, `transcribing`, or `null` when done), and `progress` (0-100%).
3. `GET /v1/audio/transcriptions/jobs/{job_id}/result` — Get the full result once completed.

```bash
python test_jobs.py audio/your_audio_file.mp3
```

### Model Management

- `GET /v1/models` — List available models (OpenAI format)
- `GET /models/list` — List currently loaded models (in VRAM)
- `POST /models/load` — Pre-load a model
- `POST /models/unload` — Free VRAM

---

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VOXHUB_MODEL` | `whisper:turbo` | Default transcription model |
| `VOXHUB_VAD` | `pyannote` | VAD mode: `silero`, `pyannote`, `hybrid`, `none` |
| `VOXHUB_DIARIZE` | `true` | Enable speaker diarization |
| `VOXHUB_SILERO_THRESHOLD` | `0.35` | Silero gate sensitivity (hybrid mode) |
| `VOXHUB_OVERRIDE_THRESHOLD` | `0.8` | Confidence override cutoff (hybrid mode) |
| `VOXHUB_API_KEY` | — | Optional API authentication |
| `VOXHUB_DEVICE` | `auto` | Hardware: `auto`, `cuda`, `rocm`, `cpu` |
| `HF_TOKEN` | — | Required for Pyannote/hybrid VAD and gated models |

> **Tip — Language Detection**: Whisper models auto-detect the spoken language by default. If you're getting translated output (e.g., English text for French audio), set the `language` parameter explicitly in your request.

### CLI Flags

```bash
python main.py audio.mp3 \
  --model whisper:turbo,voxtral:mini-4b \
  --vad hybrid \
  --diarize \
  --silero-threshold 0.35 \
  --override-threshold 0.8 \
  --precision fp16 \
  --benchmark
```

---

## Benchmarking

Benchmarking is available as an optional mode for profiling model performance.

```bash
# Benchmark specific models
python main.py audio/test.mp3 --model voxtral:mini-4b,whisper:turbo --benchmark

# Benchmark all registered models
python main.py audio/test.mp3 --model all --benchmark
```

When `--benchmark` is enabled, performance metrics are saved to `outputs/benchmarks.json`: RTF (Real-Time Factor), peak VRAM usage, and latency breakdown (model loading vs. VAD vs. transcription).

---

## Architecture

```mermaid
graph TD
    subgraph Input
        A[Audio File: .mp3, .wav, .m4a] --> B[FFmpeg Load & Normalize]
    end

    subgraph "VAD & Segmentation"
        B --> C{VAD Mode}
        C -->|silero| D[Silero VAD]
        C -->|pyannote| E[Pyannote Diarization]
        C -->|hybrid| F[Silero Gate + Pyannote Refiner]
        C -->|none| G[Full Audio Stream]

        D & E & F & G --> H[Speech Segments]
        H -.->|Cache| I[(.cache/vad)]
    end

    subgraph "ASR Registry"
        H --> J{Model Factory}
        J -->|voxtral| K[VoxtralTranscriber]
        J -->|whisper| L[WhisperTranscriber]
        J -->|moonshine| M[MoonshineTranscriber]
        J -->|canary| N[CanaryTranscriber]
    end

    subgraph "Output"
        K & L & M & N --> O[Unified Format Engine]
        O --> P[JSON / MD / TXT / SRT / VTT]
        O -.-> Q[Benchmark Metrics]
    end
```

---

## Project Structure

```
api/                  FastAPI server, routers, formatters
  routers/            Endpoint handlers (transcriptions, models, health)
  config.py           Server configuration (Pydantic Settings)
  transcriber.py      Async transcription service with job management
core/
  vad.py              Unified VAD orchestrator (Silero, Pyannote, Hybrid)
  diarize.py          Pyannote speaker diarization
  registry.py         Model factory and YAML-based registry
  transcribe*.py      Engine-specific ASR backends
  audio.py            Audio loading (FFmpeg + soundfile)
  cache.py            Persistent VAD segment caching
  benchmark.py        Performance analytics engine
  format.py           Output formatters (JSON, MD, TXT, SRT)
models.yaml           Model registry configuration
server.py             FastAPI app factory + uvicorn entry point
main.py               CLI entry point
docker-compose.yaml   Docker services (Spark, ROCm, CPU, API)
```
