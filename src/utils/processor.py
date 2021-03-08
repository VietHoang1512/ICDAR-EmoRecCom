"""
    Data processing module
"""


import math
import os
import re

from tqdm import tqdm
from utils.constant import *

tqdm.pandas()


def process_emotion_polarity(df, prefix="prob_"):
    """
    Exploding the emotion column of data

    Args:
        df (DataFrame): data
        prefix (str, optional): prefix of exploded columns. Defaults to "prob_".

    Returns:
        DataFrame: Processed dataframe
    """
    df = df.copy()
    for emotion in EMOTIONS:
        df[prefix + emotion] = 0
    for index, row in tqdm(df.iterrows(), desc="Processing emotion polarity", total=len(df)):
        emotion_polarity = eval(row["emotion_polarity"])
        for emotion, prob in emotion_polarity.items():
            assert emotion in EMOTIONS, f"Invalid emotion {emotion}"
            df.loc[index, prefix + emotion] = prob
    df = df.drop(columns=["emotion_polarity"])
    return df


def process_dialog(df, lower=True, text_separator=" "):
    """
    Simple dialog processing

    Args:
        df (DataFrame): data
        lower (bool, optional): whether lowercasing the text. Defaults to True.
        text_separator (str, optional): separator used in joining conversations. Defaults to " ".
    """

    def text_normalize(text: str):
        """
        Simple text pre-processing

        Args:
            text (str): raw text

        Returns:
            str: processed text
        """
        text = re.sub(" +", " ", text)
        text = re.sub(" ' s", "'s", text)
        text = re.sub("n ' t", "n't ", text)
        text = re.sub(" ' ll", "'ll", text)
        text = re.sub(" id ", " I'd ", text)
        text = re.sub(" shes ", " she's ", text)
        text = re.sub(" hes ", " he's ", text)
        text = re.sub("doesnt", "doesn't", text)
        text = re.sub("dont", "don't", text)
        text = re.sub(" ' ve", "'ve", text)
        text = re.sub("i ' m", "I'm", text)
        text = re.sub(" ' re", "'re", text)
        text = re.sub(" ive", " I've", text)
        text = re.sub("in ' ", "ing ", text)
        text = re.sub(" +", " ", text)
        text = text.strip()
        if lower:
            text = text.lower()
        return text

    def join_conversation(conversation):
        """
        Join conversations with text separator
        Args:
            conversation (list): dialogues in a comic frame

        Returns:
            str: a single paragraph
        """
        # TODO: handle missing values
        conversation = [item for item in conversation if (isinstance(item, str) or not math.isnan(item))]
        text = text_separator.join(conversation)
        text = text_normalize(text)
        return text

    df = df.copy()
    tqdm.pandas(desc="Merging conversations")
    df["text"] = df["dialog"].progress_apply(join_conversation)
    df = df.drop(columns=["dialog"])
    return df


def add_file_path(df, img_dir, gcs_ds_path):
    """
    Add file path to image filenames

    Args:
        df (DataFrame): data
        img_dir (str): path to the images directory
        gcs_ds_path (str): path to the GCP bucket # FIXME

    Returns:
        str: data with the image file paths
    """
    if gcs_ds_path:
        img_dir = os.path.join(gcs_ds_path, os.path.basename(img_dir))

    def get_file_path(fn):
        fp = os.path.join(img_dir, f"{fn}.jpg")
        if not gcs_ds_path:  # use local storage
            assert os.path.isfile(fp), f"{fp} not found"
        return fp

    df = df.copy()
    tqdm.pandas(desc="Adding file path information")
    df["file_path"] = df["image_id"].progress_apply(get_file_path)
    return df
