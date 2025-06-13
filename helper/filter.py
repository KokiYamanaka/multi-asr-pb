# ========================================
# SECTION: filtering algorihtm (spectral gate) - definitions
# ========================================
import numpy as np
import noisereduce as nr
from .audio_io import AudioStats 

class AudioFilter:
    @staticmethod
    def denoise(y: np.ndarray, sr: int) -> np.ndarray:
        """
        Applies noise reduction to the input waveform using noisereduce.

        Args:
            y (np.ndarray): Input audio waveform.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Denoised audio.
        """
        return nr.reduce_noise(y=y, sr=sr)
    
# ========================================
# SECTION: filtering algorihtm  - process 
# ========================================
def denoise_multiple_audio_files(audio_data: list) -> list:
    """
    Applies noise reduction to a list of audio dicts using AudioFilter.denoise,
    and recomputes stats using AudioStats.

    Args:
        audio_data (list): List of dicts containing at minimum:
            - 'name': str
            - 'y': np.ndarray
            - 'sr': int

    Returns:
        list: List of updated dicts with denoised 'y' and recomputed stats.
    """
    denoised_audios = []
    for audio in audio_data:
        y_denoised = AudioFilter.denoise(audio['y'], audio['sr'])

        denoised_audios.append({
            "name": audio["name"] + " (denoised)",
            "y": y_denoised,
            "sr": audio["sr"],
            "duration": AudioStats.duration(y_denoised, audio["sr"]),
            "rms": AudioStats.rms(y_denoised),
            "zero_crossing_rate": AudioStats.zero_crossing_rate(y_denoised),
            "estimated_snr": AudioStats.estimated_snr(y_denoised)
        })

    return denoised_audios
