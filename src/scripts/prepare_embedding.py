import argparse
import os
import pickle

import numpy as np


def get_coefs(word: str, *arr):
    """Get word-embedding line by line"""
    embedding_vector = np.asarray(arr, dtype="float32")
    return word, embedding_vector


def cache_embedding(path: str):
    """Load static embedding line by line and then save to pkl file

    Args:
        path (str): path to the static word embedding
    """
    emb_dir = os.path.dirname(path)
    file_name = os.path.basename(path)
    pickle_fn = os.path.splitext(file_name)[0] + ".pkl"
    pickle_path = os.path.join(emb_dir, pickle_fn)
    embeddings = dict(get_coefs(*o.rstrip().rsplit(" ")) for o in open(path, encoding="utf8"))
    pickle.dump(embeddings, open(pickle_path, "wb"))
    print("Cached word embedding to", pickle_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache static word embedding with pickle")
    parser.add_argument("-i", "--input", type=str, help="Path to the static word embedding")
    args = parser.parse_args()
    cache_embedding(args.input)
