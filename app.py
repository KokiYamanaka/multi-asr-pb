# app.py
from helper.audio_io import load_multiple_audio_files, AudioVisualizer
from helper.filter import denoise_multiple_audio_files, AudioFilter
from helper.transcribe import transcribe_all_audio_files
from helper.label import load_ground_truth_text, render_ground_truth_text
from helper.normalize import normalize_transcripts, normalize_text
from helper.metrics import compute_wer_per_item
# ========================================
# SECTION: render definitions
# ========================================
import pandas as pd
import streamlit as st
from typing import Dict
import numpy as np

def display_audio_summary_table(audio_data):
    """
    Display a compact summary table of audio analysis results using Streamlit.
    
    Args:
        audio_data (List[Dict]): List of dicts, each containing:
            - 'name': str, file name
            - 'sr': int, sample rate
            - 'y': np.ndarray, waveform
            - 'duration': float
            - 'rms': float
            - 'zero_crossing_rate': float
            - 'estimated_snr': float
    """
    st.subheader("📊 Audio Basic Stat")
    rows = []
    for audio in audio_data:
        rows.append({
            "File": audio["name"],
            "SR": audio["sr"],
            "Samples": audio["y"].shape[0],
            "Duration (s)": round(audio["duration"], 2),
            "RMS": round(audio["rms"], 6),
            "ZCR": round(audio["zero_crossing_rate"], 6),
            "SNR (dB)": round(audio["estimated_snr"], 2)
        })

    df = pd.DataFrame(rows)
    st.table(df)


def render_audio_plots(audio_data):
    """
    Displays each audio file in a column, with waveform (top) and spectrogram (bottom).

    Args:
        audio_data (List[Dict]): List of dicts with keys:
            - 'name': str
            - 'y': np.ndarray
            - 'sr': int
    """
     

    columns = st.columns(len(audio_data))  # one column per audio file

    for i, audio in enumerate(audio_data):
        with columns[i]:
            st.markdown(f"**🎧 {audio['name']}**")
            st.pyplot(AudioVisualizer.plot_waveform(audio['y'], audio['sr']))
            st.pyplot(AudioVisualizer.plot_spectrogram(audio['y'], audio['sr']))

def denoise_audio(audio: Dict) -> Dict:
    """
    Applies AudioFilter.denoise() to a single audio dictionary.

    Args:
        audio (dict): A dictionary with keys:
            - 'y': np.ndarray, the waveform
            - 'sr': int, the sample rate
            - 'name': (optional) file name or metadata

    Returns:
        dict: Same as input, with 'y' replaced by denoised version.
    """
    y_denoised = AudioFilter.denoise(audio['y'], audio['sr'])

    return {
        **audio,
        "y": y_denoised
    }
 



def show_table(audio_data, ground_truth_text: str):
    """
    Display transcripts and WER against ground truth in an adjustable dataframe.
    """
    rows = []

    for item in audio_data:
        transcript = item.get("transcript", "").strip()
        wer_value = round(item.get("wer", 0.0), 4)

        rows.append({
            "File": item["name"],
            "Transcript": transcript,
            "WER": wer_value
        })

    # Add ground truth row
    rows.append({
        "File": "✅ Ground Truth",
        "Transcript": ground_truth_text,
        "WER": None
    })

    df = pd.DataFrame(rows)

    st.subheader("🪞 Transcription Comparison (Full View)")
    st.dataframe(df, use_container_width=True)



# ========================================
# SECTION: trigger render
# ========================================
st.title("Multi-Mic ASR Demo")

# ground truth texts (single .txt file)
ground_truth_text = load_ground_truth_text()
ground_truth_text = normalize_text(ground_truth_text)

# file uploader for 3 audio files
files = st.file_uploader("Upload 3 audio files", type=["wav", "mp3"], accept_multiple_files=True)


if len(files) == 3:
    st.success("3 files uploaded!")

    # load raw audio files 
    audio_data = load_multiple_audio_files(files)
     
    # Compute basic stats for each audio file 
    display_audio_summary_table(audio_data)
    render_audio_plots(audio_data)

    # Denoise the audio files 
    denoised_audio_data = denoise_multiple_audio_files(audio_data)
    display_audio_summary_table(denoised_audio_data)
    render_audio_plots(denoised_audio_data)

    # Transcribe the denoised audio files
    transcribed_data = transcribe_all_audio_files(denoised_audio_data)
    # normalize the transcripts
    normalized_data = normalize_transcripts(transcribed_data)
    # inside normalized_data : a list of py dict
    #sample 
    #{'name': 'ayush.wav (denoised)', 'y': array([-8.85716714e-20, -4.54632941e-20,  6.87184119e-21, ...,
    #    -5.55595045e-06, -6.80034504e-06, -8.33937584e-06], shape=(240000,)), 'sr': 16000, 'duration': 15.0, 'rms': 0.010970761068165302, 'zero_crossing_rate': 0.1898820628997868, 'estimated_snr': np.float64(18.701552270732154), 'transcript': 'so the mvp scope is simple voice 1st interface task management right yeah usually you can see things like we might need audacity or schedule zoom with', 'wer': 0.3548387096774194}
    

    # compute wer 
    wer_computed_data = compute_wer_per_item(normalized_data,ground_truth_text)

     # Show the table with the normalized transcripts
    show_table(wer_computed_data, ground_truth_text)



 