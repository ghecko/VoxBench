import os
import logging
import torch
import numpy as np
from typing import List, Dict, Optional
from core.diarize import DiarizationAnalyzer

logger = logging.getLogger(__name__)


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

    def detect_with_probabilities(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments AND return per-segment confidence scores.
        Used by HybridVAD to make override decisions.
        Returns [{"start": float_s, "end": float_s, "probability": float}]
        """
        wav = torch.from_numpy(audio.copy())

        # Get raw frame-level speech probabilities from Silero
        from silero_vad import get_speech_timestamps
        timestamps = get_speech_timestamps(
            wav, self.model,
            sampling_rate=sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )

        # Re-run the model on each detected segment to get its peak probability.
        # Silero operates on 512-sample (32ms @ 16kHz) frames.
        frame_size = 512
        results = []
        for t in timestamps:
            start_samp = int(t["start"] * sampling_rate)
            end_samp = int(t["end"] * sampling_rate)
            segment_wav = wav[start_samp:end_samp]

            # Compute frame-level probabilities for this segment
            max_prob = 0.0
            self.model.reset_states()
            for i in range(0, len(segment_wav), frame_size):
                frame = segment_wav[i:i + frame_size]
                if len(frame) < frame_size:
                    frame = torch.nn.functional.pad(frame, (0, frame_size - len(frame)))
                prob = self.model(frame.unsqueeze(0), sampling_rate).item()
                max_prob = max(max_prob, prob)

            results.append({
                "start": t["start"],
                "end": t["end"],
                "probability": round(max_prob, 4),
            })

        return results


class HybridVAD:
    """
    Hybrid VAD Pipeline: Silero (gate) + Pyannote (refiner).

    Strategy ("Refined Gating"):
        1. Silero runs first with a sensitive threshold to capture all potential speech.
        2. Only Silero-detected regions are passed to Pyannote for fine-grained
           segmentation and optional speaker diarization.
        3. If Pyannote rejects a Silero segment but Silero's confidence was above
           the override threshold, the segment is kept anyway (safety net).

    This gives you Silero's recall with Pyannote's precision, while the override
    threshold acts as a safety net for high-confidence speech that Pyannote
    might drop (e.g. very short utterances, overlapping speakers).

    Design note — Pyannote on full audio vs. gated regions:
        Pyannote's diarization model runs on the FULL audio, not just the Silero-gated
        regions. This is intentional: Pyannote's speaker embedding and clustering
        require global context to assign consistent speaker labels across the entire
        file. Feeding it isolated chunks would produce fragmented, inconsistent speaker
        IDs (e.g., the same person labeled SPEAKER_00 in one chunk and SPEAKER_02 in
        another).

        If you don't need diarization and only want precise VAD boundaries, a future
        "gated-only" variant could pass only Silero's regions to Pyannote's
        segmentation model (not the full diarization pipeline) for a significant
        speed-up. This is not yet implemented.

    Args:
        silero_threshold:  Silero detection sensitivity. Lower = more sensitive.
                           Default 0.35 (more aggressive than standalone default of 0.5).
        override_threshold: Silero probability above which a segment is kept even
                            if Pyannote disagrees. Default 0.8.
        hf_token:          HuggingFace token for Pyannote authentication.
    """

    def __init__(
        self,
        silero_threshold: float = 0.35,
        override_threshold: float = 0.8,
        hf_token: Optional[str] = None,
    ):
        self.silero_threshold = silero_threshold
        self.override_threshold = override_threshold
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._silero: Optional[SileroVAD] = None
        self._pyannote: Optional[DiarizationAnalyzer] = None

    @property
    def silero(self) -> SileroVAD:
        if self._silero is None:
            self._silero = SileroVAD(
                threshold=self.silero_threshold,
                min_speech_duration_ms=200,   # More permissive for gating
                min_silence_duration_ms=300,
            )
        return self._silero

    @property
    def pyannote(self) -> DiarizationAnalyzer:
        if self._pyannote is None:
            self._pyannote = DiarizationAnalyzer(auth_token=self.hf_token)
        return self._pyannote

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000, **kwargs) -> List[Dict]:
        """
        Run the hybrid pipeline.

        Returns a list of segments. If diarize=True, segments include "speaker" labels
        from Pyannote. Override segments (Silero-only) get speaker="SPEAKER_OVERRIDE".
        """
        diarize = kwargs.get("diarize", False)

        # --- Step 1: Silero gate (sensitive) ---
        logger.info(
            f"[HybridVAD] Step 1/3: Silero gate (threshold={self.silero_threshold})"
        )
        silero_segments = self.silero.detect_with_probabilities(audio, sampling_rate)
        logger.info(f"[HybridVAD]   -> {len(silero_segments)} candidate segments")

        if not silero_segments:
            return []

        # --- Step 2: Pyannote refiner on gated regions ---
        logger.info("[HybridVAD] Step 2/3: Pyannote refinement")
        pyannote_kwargs = {}
        if diarize:
            for key in ("num_speakers", "min_speakers", "max_speakers"):
                if kwargs.get(key) is not None:
                    pyannote_kwargs[key] = kwargs[key]

        pyannote_segments = self.pyannote.diarize(
            audio, sampling_rate=sampling_rate, **pyannote_kwargs
        )
        logger.info(f"[HybridVAD]   -> {len(pyannote_segments)} refined segments")

        # --- Step 3: Reconcile with safety-net override ---
        logger.info(
            f"[HybridVAD] Step 3/3: Reconciling (override_threshold={self.override_threshold})"
        )
        final_segments = list(pyannote_segments)  # Start with Pyannote's output

        overrides_added = 0
        for s_seg in silero_segments:
            # Check if Pyannote already covers this region (>50% overlap)
            covered = self._is_covered(s_seg, pyannote_segments)
            if not covered and s_seg["probability"] >= self.override_threshold:
                # Safety net: Silero is very confident but Pyannote missed it
                final_segments.append({
                    "start": s_seg["start"],
                    "end": s_seg["end"],
                    "speaker": "SPEAKER_OVERRIDE" if diarize else "SPEAKER_00",
                })
                overrides_added += 1

        if overrides_added:
            logger.info(
                f"[HybridVAD]   -> {overrides_added} Silero override(s) added"
            )

        # Sort by start time
        final_segments.sort(key=lambda s: s["start"])

        # Strip speaker labels if diarization was not requested
        if not diarize:
            final_segments = [
                {"start": s["start"], "end": s["end"], "speaker": s.get("speaker", "SPEAKER_00")}
                for s in final_segments
            ]

        logger.info(f"[HybridVAD] Final: {len(final_segments)} segments")
        return final_segments

    @staticmethod
    def _is_covered(silero_seg: Dict, pyannote_segments: List[Dict], min_overlap: float = 0.5) -> bool:
        """
        Check whether a Silero segment is sufficiently covered by any Pyannote segment.
        Coverage is measured as the fraction of the Silero segment's duration that
        overlaps with Pyannote output.
        """
        s_start, s_end = silero_seg["start"], silero_seg["end"]
        s_dur = s_end - s_start
        if s_dur <= 0:
            return True  # Degenerate segment, skip

        total_overlap = 0.0
        for p_seg in pyannote_segments:
            overlap_start = max(s_start, p_seg["start"])
            overlap_end = min(s_end, p_seg["end"])
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start

        return (total_overlap / s_dur) >= min_overlap


class UnifiedVAD:
    """
    Unified VAD Orchestrator.

    Supported modes:
        - "silero":   Fast, lightweight Silero VAD. Good default for speed.
        - "pyannote": High-quality Pyannote segmentation + optional diarization.
        - "hybrid":   Silero as a sensitive gate, Pyannote as a refiner, with a
                      confidence-based safety net. Best balance of recall and precision.
        - "none":     No VAD — treat the entire audio as one segment.

    See HybridVAD docstring for details on the Refined Gating strategy.
    """
    def __init__(self, mode: str = "silero", hf_token: Optional[str] = None,
                 silero_threshold: float = 0.35, override_threshold: float = 0.8):
        self.mode = mode
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.silero_threshold = silero_threshold
        self.override_threshold = override_threshold
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
                segments = self.vad_model.diarize(audio, sampling_rate=sampling_rate)
                return segments

        elif self.mode == "hybrid":
            if not self.vad_model:
                self.vad_model = HybridVAD(
                    silero_threshold=self.silero_threshold,
                    override_threshold=self.override_threshold,
                    hf_token=self.hf_token,
                )
            return self.vad_model.detect(audio, sampling_rate, **kwargs)

        elif self.mode == "none":
            # Pass full audio as a single segment
            duration = len(audio) / sampling_rate
            return [{"start": 0.0, "end": duration}]

        else:
            raise ValueError(f"Unknown VAD mode: {self.mode}")
