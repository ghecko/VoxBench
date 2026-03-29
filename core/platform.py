"""
Platform detection utilities for Voxtral-Transcribe.

Detects whether we're running on:
  - blackwell  : NVIDIA Blackwell GPU (GB10 / B200 etc.) — supports NVFP4
  - cuda       : NVIDIA GPU (pre-Blackwell) — uses bitsandbytes quantization
  - rocm       : AMD ROCm GPU — uses bitsandbytes quantization
  - cpu        : CPU-only fallback
"""

import os
import torch


def detect_platform() -> str:
    """
    Auto-detect the compute platform at runtime.
    Returns one of: "blackwell", "cuda", "rocm", "cpu".
    """
    # ROCm exposes CUDA-like APIs through HIP, so check ROCm indicators first
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")
    torch_hip = getattr(torch.version, "hip", None)

    if torch_hip or rocm_home or hip_visible is not None:
        if torch.cuda.is_available():
            return "rocm"
        return "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Blackwell is compute capability 10.x (sm_100)
        if props.major >= 10:
            return "blackwell"
        return "cuda"

    return "cpu"


def get_optimal_device_map(platform: str) -> str:
    """
    Return the best device_map strategy for the detected platform.

    - blackwell / cuda: "cuda:0" — skip Accelerate's profiling pass.
      On GB10 the 128 GB unified memory makes multi-device splitting pointless,
      and on single-GPU CUDA boxes it avoids the extra overhead too.
    - rocm: "auto" — let Accelerate handle HIP device placement.
    - cpu: "cpu".
    """
    if platform in ("blackwell", "cuda"):
        return "cuda:0"
    if platform == "rocm":
        return "auto"
    return "cpu"


def get_torch_dtype(platform: str, precision: str) -> torch.dtype:
    """
    Return the torch dtype to use when loading the model.

    - NVFP4 quantization on Blackwell requires bf16 as the base dtype.
    - Auto mode uses "auto" to let the model config decide (important for
      pre-quantized checkpoints that store their own dtype).
    - Everything else uses fp16 (or the quantiser's own dtype).
    """
    if precision == "auto":
        return "auto"  # type: ignore[return-value]  — transformers accepts "auto"
    if platform == "blackwell" and precision == "nvfp4":
        return torch.bfloat16
    return torch.float16


def supports_nvfp4(platform: str) -> bool:
    """True if the platform has native NVFP4 tensor-core support."""
    return platform == "blackwell"


def platform_summary(platform: str) -> str:
    """Human-readable one-liner for logs."""
    labels = {
        "blackwell": "NVIDIA Blackwell (NVFP4-capable)",
        "cuda": "NVIDIA CUDA GPU",
        "rocm": "AMD ROCm GPU",
        "cpu": "CPU-only",
    }
    return labels.get(platform, platform)
