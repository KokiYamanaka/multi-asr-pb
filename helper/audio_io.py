
import numpy as np
import librosa
from io import BytesIO
import soundfile as sf

def load_audio_file(uploaded_file, sr: int = 16000) -> tuple[np.ndarray, int]:
    file_bytes = uploaded_file.read()
    audio_file = BytesIO(file_bytes)
    y, sr_actual = sf.read(audio_file)
    if sr_actual != sr:
        y = librosa.resample(y.T, orig_sr=sr_actual, target_sr=sr).T
    return y, sr

def load_multiple_audio_files(files):
    audios = []
    for f in files:
        y, sr = load_audio_file(f)
        audios.append({
            "name": f.name,
            "y": y,
            "sr": sr,
            "duration": AudioStats.duration(y, sr),
            "rms": AudioStats.rms(y),
            "zero_crossing_rate": AudioStats.zero_crossing_rate(y),
            "estimated_snr": AudioStats.estimated_snr(y)
        })
    return audios

# ========================================
# SECTION: other metadata 
# ========================================
import numpy as np
import librosa

class AudioStats:
    @staticmethod
    def duration(y: np.ndarray, sr: int) -> float:
        """Returns duration of the audio in seconds."""
        return librosa.get_duration(y=y, sr=sr)

    @staticmethod
    def rms(y: np.ndarray) -> float:
        """Returns root mean square energy."""
        return float(np.mean(librosa.feature.rms(y=y)))

    @staticmethod
    def zero_crossing_rate(y: np.ndarray) -> float:
        """Returns average zero-crossing rate."""
        return float(np.mean(librosa.feature.zero_crossing_rate(y)))

    @staticmethod
    def estimated_snr(y: np.ndarray) -> float:
        """
        Returns a crude estimated SNR:
        signal power vs. 'silence' power (defined as samples with amplitude < 0.01).
        """
        signal_power = np.mean(y**2)
        silence_power = np.mean((y[np.abs(y) < 0.01])**2) + 1e-10  # avoid div by zero
        return 10 * np.log10(signal_power / silence_power)

# ========================================
# SECTION: plottings 
# ========================================
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

class AudioVisualizer:
    @staticmethod
    def plot_waveform(y: np.ndarray, sr: int):
        """Returns a compact waveform figure."""
        fig, ax = plt.subplots(figsize=(4, 1.2))  # smaller figure
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=6)
        return fig

    @staticmethod
    def plot_spectrogram(y: np.ndarray, sr: int):
        """Returns a compact mel spectrogram figure."""
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(4, 1.8))  # smaller figure
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title("Mel Spectrogram", fontsize=10)
        ax.tick_params(labelsize=6)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return fig