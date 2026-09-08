"""Lab 3B: NER label alignment."""


def align_labels(word_ids, word_labels):
    """
    Align word-level NER labels with tokenizer subwords.

    Special tokens and continuation subwords receive -100
    so they are ignored during training.
    """

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)

        elif word_id != previous_word_id:
            aligned_labels.append(word_labels[word_id])

        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels