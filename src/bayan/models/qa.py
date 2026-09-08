"""Lab 3B: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    """
    Find the best valid answer span or return an honest null answer.
    """

    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    start_indices = np.argsort(start_logits)[-top_k:][::-1]
    end_indices = np.argsort(end_logits)[-top_k:][::-1]

    best_candidate = None
    best_score = float("-inf")

    for start_index in start_indices:
        for end_index in end_indices:
            start_index = int(start_index)
            end_index = int(end_index)

            if start_index >= len(offsets):
                continue

            if end_index >= len(offsets):
                continue

            if offsets[start_index] is None:
                continue

            if offsets[end_index] is None:
                continue

            if end_index < start_index:
                continue

            answer_length = end_index - start_index + 1

            if answer_length > max_answer_len:
                continue

            start_char = offsets[start_index][0]
            end_char = offsets[end_index][1]

            if end_char <= start_char:
                continue

            score = float(
                start_logits[start_index]
                + end_logits[end_index]
            )

            if score > best_score:
                best_score = score

                best_candidate = {
                    "answer": (start_char, end_char),
                    "start_token": start_index,
                    "end_token": end_index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "score": score,
                }

    if best_candidate is None:
        return {
            "answer": None,
            "score": float(null_score),
        }

    null_difference = float(null_score) - best_score

    if null_difference > null_threshold:
        return {
            "answer": None,
            "score": float(null_score),
        }

    return best_candidate