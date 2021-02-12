"""
    This file contain the data directory structure and format
    https://drive.google.com/file/d/1OUd7dQybiioKMu7NXtWxIITdun8SaaUX/view
"""

TRAIN_IMG_DIR = "train"
TEST_IMG_DIR = "test"
TRAIN_LABELS = "train_5_folds.csv"
TRAIN_POLARITY = "train_emotion_polarity.csv"
TRAIN_SCRIPT = "train_transcriptions.json"
TEST_SCRIPT = "test_transcriptions.json"
SAMPLE_SUBMISSION = "results.csv"
OUTPUT_DIR = "outputs"

GCS_DS_PATH = None
# GCS_DS_PATH = 'gs://kds-1dbcc2e911f8e605b6311dc9c69ebd6ecb338b28ee58c2088ee88e65'

ALL_COLS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "other"]
EMOTIONS = ["neutral", "fear", "surprise", "angry", "happy", "sad", "disgust"]
IMAGE_MODELS = ["efn_b0", "efn_b1", "efn_b2", "efn_b3", "efn_b4", "efn_b5", "efn_b6", "efn_b7"]
