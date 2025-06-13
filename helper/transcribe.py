import os
import soundfile as sf
from typing import List, Dict
import numpy as np


# import os
# from imageio_ffmpeg import get_ffmpeg_exe


# # ─── Bundle ffmpeg into PATH ─────────────────────────────────────────
# ffmpeg_path = get_ffmpeg_exe()                      # locate the downloaded binary
# ffmpeg_dir  = os.path.dirname(ffmpeg_path)          # its folder
# os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
# ─────────────────────────────────────────────────────────────────────
import whisper

class WhisperTranscriber:
    def __init__(self, model_size: str = "base", save_dir: str = "dump"):
        self.model = whisper.load_model(model_size)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def transcribe_single(self, y: np.ndarray, sr: int, filename: str) -> str:
        """
        Saves audio and transcribes it. Also writes transcript to .txt file.

        Args:
            y (np.ndarray): Audio waveform
            sr (int): Sample rate
            filename (str): e.g., "mic1.wav"

        Returns:
            str: Transcribed text
        """
        audio_path = os.path.join(self.save_dir, filename)
        sf.write(audio_path, y, sr)

        result = self.model.transcribe(audio_path)

        transcript_path = os.path.join(self.save_dir, filename.replace(".wav", ".txt"))
        with open(transcript_path, "w") as f:
            f.write(result["text"])

        return result["text"]

    def transcribe_multiple(self, audio_data: List[Dict]) -> List[Dict]:
        """
        Transcribes multiple audio entries, saving outputs for each.

        Args:
            audio_data (List[Dict]): List of dicts from load_multiple_audio_files

        Returns:
            List[Dict]: Each dict includes original metadata + 'transcript'
        """
        results = []
        for audio in audio_data:
            filename = audio["name"].replace(" ", "_")
            if not filename.endswith(".wav"):
                filename = filename.rsplit(".", 1)[0] + ".wav"

            transcript = self.transcribe_single(audio["y"], audio["sr"], filename)
            results.append({**audio, "transcript": transcript})
        return results

def transcribe_all_audio_files(audio_data, model_size="base", save_dir="dump"):
    """
    Transcribes a list of audio files using Whisper and returns updated dicts with transcripts.

    Args:
        audio_data (List[Dict]): List of dicts with keys: 'name', 'y', 'sr'
        model_size (str): Whisper model size (e.g., 'base', 'tiny', 'small', etc.)
        save_dir (str): Folder to save .wav and .txt outputs

    Returns:
        List[Dict]: Original dicts with added 'transcript' key
    """
    transcriber = WhisperTranscriber(model_size=model_size, save_dir=save_dir)
    transcribed_data = transcriber.transcribe_multiple(audio_data)

    for item in transcribed_data:
        print(f"{item['name']} → {item['transcript'][:100]}...")

    return transcribed_data