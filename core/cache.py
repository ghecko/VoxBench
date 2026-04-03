import os
import json
import hashlib
from typing import List, Dict, Optional

class VADCache:
    """
    Caching system for VAD segments to avoid redundant processing.
    """
    def __init__(self, cache_dir: str = ".cache/vad"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, audio_data: bytes, vad_params: Dict) -> str:
        """Calculate a unique cache key from audio hashing and params."""
        # Hash the audio data (first and last 1MB for speed? No, full hash is safer)
        audio_hash = hashlib.md5(audio_data).hexdigest()
        
        # Hash the parameters
        params_str = json.dumps(vad_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        # Final key
        cache_key = f"{audio_hash}_{params_hash}"
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def load(self, audio_data: bytes, vad_params: Dict) -> Optional[List[Dict]]:
        """Load segments from cache if they exist."""
        path = self.get_cache_path(audio_data, vad_params)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save(self, audio_data: bytes, vad_params: Dict, segments: List[Dict]):
        """Save segments to cache."""
        path = self.get_cache_path(audio_data, vad_params)
        try:
            with open(path, "w") as f:
                json.dump(segments, f, indent=4)
        except Exception:
            pass
