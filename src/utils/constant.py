"""
    This file contains the data directory structure and files format
    https://drive.google.com/file/d/1bofnf_jiELtKuSQfU8lXy6zDKutWUgzs/view?usp=sharing
"""

IMAGE_DIR = "images"
TRAIN_LABELS = "train_5_folds.csv"
TRAIN_POLARITY = "additional_infor:train_emotion_polarity.csv"
TRAIN_SCRIPT = "train_transcriptions.json"
TEST_SCRIPT = "transcriptions.json"
SAMPLE_SUBMISSION = "results.csv"

GCS_DS_PATH = None
# GCS_DS_PATH = 'gs://kds-1dbcc2e911f8e605b6311dc9c69ebd6ecb338b28ee58c2088ee88e65'

ALL_COLS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "other"]
EMOTIONS = ["neutral", "fear", "surprise", "angry", "happy", "sad", "disgust"]
IMAGE_MODELS = ["efn-b0", "efn-b1", "efn-b2", "efn-b3", "efn-b4", "efn-b5", "efn-b6", "efn-b7"]
WORD_EMBEDDING_MODELS = ["glove.840B.300d", "wiki.en.vec", "crawl-300d-2M.vec", "wiki-news-300d-1M.vec"]
LOAD_ARGS = [
    "image_model",
    "bert_model",
    "word_embedding",
    "max_vocab",
    "max_word",
    "image_size",
    "max_len",
    "lower",
    "text_separator",
    "n_hiddens",
    "drop_rate",
    "target_cols",
]
