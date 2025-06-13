# ========================================
# SECTION: render definitions 
# ========================================
from typing import List, Dict
import streamlit as st

def load_ground_truth_text() -> str:
    """
    Upload a single .txt file and return its raw content as a string.
    
    Returns:
        str: Raw ground truth text, or empty string if no file uploaded.
    """
    file = st.file_uploader(
        "Upload ground truth single text file (.txt only)", 
        type=["txt"]
    )

    if not file:
        return ""

    return file.read().decode("utf-8").strip()

def render_ground_truth_text(text: str):
    st.markdown("### ✅ Ground Truth Preview")
    st.text_area("All Ground Truth Texts", text, height=300)

