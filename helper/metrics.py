import jiwer

def compute_wer_per_item(data: list[dict], ground_truth: str) -> list[dict]:
    """
    Adds a 'wer' score to each item in the list comparing its 'transcript'
    to the provided ground truth string.

    Args:
        data (list of dict): Each dict must have 'name' and 'transcript'.
        ground_truth (str): The reference transcript string.

    Returns:
        list of dict: Original list with an added 'wer' field in each dict.
    """
    enriched = []

    for item in data:
        transcript = item.get("transcript", "").strip()

        # Skip if empty
        if not transcript:
            wer_score = None
        else:
            wer_score = jiwer.measures.wer(ground_truth, transcript)

        item_with_wer = item.copy()
        item_with_wer["wer"] = wer_score
        enriched.append(item_with_wer)

    return enriched