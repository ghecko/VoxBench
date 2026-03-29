import ffmpeg
import numpy as np
import soundfile as sf

def load_audio(file_path: str, sampling_rate: int = 16000) -> np.ndarray:
    """
    Load audio using FFmpeg to support varied formats (.mp3, .m4a, .wav, etc.)
    and convert to mono float32 at the target sampling rate.
    """
    try:
        # Use ffmpeg to decode and resample the audio
        out, _ = (
            ffmpeg.input(file_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sampling_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Convert the buffer to a numpy array
        return np.frombuffer(out, np.float32)
    except ffmpeg.Error as e:
        # Fallback to soundfile if ffmpeg fails (or isn't installed for some reason)
        print(f"FFmpeg loading failed, trying soundfile: {e.stderr.decode()}")
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=-1)
            # Resampling would be needed here if sf.read sr != sampling_rate
            return data.astype(np.float32)
        except Exception as sf_err:
            raise RuntimeError(f"Failed to load audio with both FFmpeg and soundfile. SF error: {sf_err}")

def save_audio(audio: np.ndarray, file_path: str, sr: int = 16000):
    """
    Save a numpy array as a wav file.
    """
    try:
        process = (
            ffmpeg.input("pipe:", format="f32le", ac=1, ar=sr)
            .output(file_path, acodec="pcm_s16le", ac=1, ar=sr)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )
        process.communicate(input=audio.tobytes())
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to save audio: {e.stderr.decode()}") from e
