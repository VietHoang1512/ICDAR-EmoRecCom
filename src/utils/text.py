import logging
import pickle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def length_plot(lengths: List[int]) -> None:
    """
    Plot the sequence length statistic

    Args:
        lengths (List): Sequence lengths (by word or character)
    Returns:
        None
    """
    plt.figure(figsize=(15, 9))
    textstr = f" Mean: {np.mean(lengths):.2f} \u00B1 {np.std(lengths):.2f} \n Max: {np.max(lengths)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    plt.text(0, 0, textstr, fontsize=14, verticalalignment="top", bbox=props)
    sns.countplot(lengths, orient="h")
    plt.show()


def get_embedding_matrix(word_index: Dict[str, Any], max_vocab: int, embedding_fp: str, lower: bool):
    """
    Generate embedding matrix with the provided word index

    Args:
        word_index (Dict[str, Any]): A dictionary map word to vector
        max_vocab (int): Maximum number of vocabulary
        embedding_fp (str): Path to the embedding weight location
        lower (bool): whether lowercase the text

    Returns:
        embedding matrix
    """
    logging.info(f"Word vocabulary size: {len(word_index)}")
    embeddings = pickle.load(open(embedding_fp, "rb"))

    if lower:
        embeddings = {word.lower(): vector for word, vector in embeddings.items()}
    # prepare embedding matrix
    num_words = min(max_vocab, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, 300))
    miss = 0
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word.lower())
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            miss += 1
    logging.warning(f"Missed {miss} in total {num_words} words")
    return embedding_matrix
