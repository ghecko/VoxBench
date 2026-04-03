import time
import json
import os
import torch
from datetime import datetime
from typing import Dict, Optional

class BenchmarkTracker:
    """
    Benchmark tracker for performance analytics.
    Tracks times (load, VAD, transcription) and RTF.
    """
    def __init__(self, model_spec: str, vad_mode: str, device: str):
        # Reset peak memory stats at the start of each model benchmark
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": model_spec,
            "vad": vad_mode,
            "device": device,
            "audio_duration_s": 0.0,
            "load_time_s": 0.0,
            "vad_time_s": 0.0,
            "transcription_time_s": 0.0,
            "total_time_s": 0.0,
            "rtf": 0.0,
            "peak_vram_gb": 0.0,
        }
        self.start_total = time.time()

    def set_duration(self, duration_s: float):
        self.metrics["audio_duration_s"] = round(duration_s, 2)

    def mark_load_done(self, start_time: float):
        self.metrics["load_time_s"] = round(time.time() - start_time, 2)

    def mark_vad_done(self, start_time: float):
        self.metrics["vad_time_s"] = round(time.time() - start_time, 2)

    def mark_transcription_done(self, start_time: float):
        self.metrics["transcription_time_s"] = round(time.time() - start_time, 2)

    def finalize(self):
        self.metrics["total_time_s"] = round(time.time() - self.start_total, 2)
        
        # Calculate Real-Time Factor (RTF)
        # RTF = Audio Duration / Processing Time
        if self.metrics["total_time_s"] > 0:
            self.metrics["rtf"] = round(self.metrics["audio_duration_s"] / self.metrics["total_time_s"], 2)
        
        # Track Peak VRAM (Reserved memory is more representative of nvidia-smi)
        if torch.cuda.is_available():
            self.metrics["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / (1024**3), 2)

    def save(self, output_path: str = "outputs/benchmarks.json"):
        """Append the benchmarks to a central JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        benchmarks = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    benchmarks = json.load(f)
            except Exception:
                pass
        
        benchmarks.append(self.metrics)
        
        with open(output_path, "w") as f:
            json.dump(benchmarks, f, indent=4)

    def print_summary(self, console):
        """Print a nice summary to the rich console."""
        console.print()
        console.print("[bold green]━━━ Benchmark Result ━━━[/bold green]")
        console.print(f"  Model         : [cyan]{self.metrics['model']}[/cyan]")
        console.print(f"  VAD Path      : [cyan]{self.metrics['vad']}[/cyan]")
        console.print(f"  RTF           : [bold green]{self.metrics['rtf']}x[/bold green] (Faster than real-time)")
        console.print(f"  Total Time    : {self.metrics['total_time_s']}s")
        console.print(f"  Peak VRAM     : {self.metrics['peak_vram_gb']} GB")
        console.print()
