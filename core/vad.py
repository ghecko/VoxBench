import os
import torch
import numpy as np
from typing import List, Dict, Optional
from core.diarize import DiarizationAnalyzer

class SileroVAD:
    """
    Silero Voice Activity Detection - Lightweight and fast.
    """
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 500):
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.model = load_silero_vad()
            self.get_speech_timestamps = get_speech_timestamps
            self.threshold = threshold
            self.min_speech_duration_ms = min_speech_duration_ms
            self.min_silence_duration_ms = min_silence_duration_ms
        except ImportError:
            raise ImportError("silero-vad not installed. Run 'pip install silero-vad'")

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments in the audio.
        Returns [{"start": float_s, "end": float_s}]
        """
        wav = torch.from_numpy(audio.copy())
        timestamps = self.get_speech_timestamps(
            wav, self.model,
            sampling_rate=sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )
        return [{"start": t["start"], "end": t["end"]} for t in timestamps]


class UnifiedVAD:
    """
    Unified VAD Orchestrator. Supports both Silero and Pyannote.
    """
    def __init__(self, mode: str = "silero", hf_token: Optional[str] = None):
        self.mode = mode
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.vad_model = None

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000, **kwargs) -> List[Dict]:
        """
        Orchestrate VAD based on the selected mode.
        """
        if self.mode == "silero":
            if not self.vad_model:
                self.vad_model = SileroVAD()
            return self.vad_model.detect(audio, sampling_rate)
        
        elif self.mode == "pyannote":
            if not self.vad_model:
                self.vad_model = DiarizationAnalyzer(auth_token=self.hf_token)
            
            # Check if we want full diarization or VAD-only
            diarize = kwargs.get("diarize", False)
            if diarize:
                return self.vad_model.diarize(
                    audio, 
                    sampling_rate=sampling_rate,
                    num_speakers=kwargs.get("num_speakers"),
                    min_speakers=kwargs.get("min_speakers"),
                    max_speakers=kwargs.get("max_speakers")
                )
            else:
                # Mock a diarize call to get segments (no labels)
                segments = self.vad_model.diarize(audio, sampling_rate=sampling_rate)
                # Group by speaker labels but ignore them for VAD-only?
                # Actually, Pyannote diarize already returns segments.
                return segments
        
        elif self.mode == "none":
            # Pass full audio as a single segment
            duration = len(audio) / sampling_rate
            return [{"start": 0.0, "end": duration}]
        
        else:
            raise ValueError(f"Unknown VAD mode: {self.mode}")
