from whisper_normalizer.english import EnglishTextNormalizer

def normalize_text(text: str) -> str:
    """
    Normalize English text using whisper-normalizer.
    
    Args:
        text (str): Input transcript text.
    
    Returns:
        str: Cleaned and normalized text.
    """
    normalizer = EnglishTextNormalizer()
    return normalizer(text)

def normalize_transcripts(audio_data: list) -> list:
    """
    Normalize the 'transcript' field in each audio data entry using EnglishTextNormalizer.
    
    Args:
        audio_data (list): List of dicts with 'transcript' field.
    
    Returns:
        list: Same list with normalized transcripts.
    """
    for entry in audio_data:
        if "transcript" in entry and isinstance(entry["transcript"], str):
            entry["transcript"] = normalize_text(entry["transcript"])
    return audio_data


 