import json
from typing import List, Dict

class OutputFormatter:
    @staticmethod
    def to_json(data: List[Dict], output_path: str):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def to_markdown(data: List[Dict], output_path: str):
        lines = ["# Transcription Report\n"]
        for entry in data:
            speaker = entry.get("speaker", "Unknown")
            start = entry.get("start", 0)
            text = entry.get("text", "")
            lines.append(f"**[{speaker}] ({start:.2f}s):** {text}\n")
        
        with open(output_path, "w") as f:
            f.writelines(lines)

    @staticmethod
    def to_txt(data: List[Dict], output_path: str):
        lines = []
        for entry in data:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{speaker}: {text}\n")
            
        with open(output_path, "w") as f:
            f.writelines(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def to_srt(data: List[Dict], output_path: str):
        lines = []
        for i, entry in enumerate(data, 1):
            start = OutputFormatter._format_srt_time(entry.get("start", 0))
            end = OutputFormatter._format_srt_time(entry.get("end", 0))
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n\n")
        
        with open(output_path, "w") as f:
            f.writelines(lines)
