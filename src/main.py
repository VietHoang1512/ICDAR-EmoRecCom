import argparse
import json
import os
import pickle
import re
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import yaml
from tensorflow.keras import backend as K
from trainer.dataset import ICDARGenerator
from trainer.model import build_model
from trainer.utils import scheduler, seed_all, select_strategy
from transformers import AutoTokenizer
from utils.constant import *
from utils.logger import custom_logger
from utils.processor import (
    add_file_path,
    process_dialog,
    process_emotion_polarity,
)
from utils.signature import print_signature
from utils.text import get_embedding_matrix

print("Using Tensorflow version:", tf.__version__)
print("Using Transformers version:", transformers.__version__)


parser = argparse.ArgumentParser(description="ICDAR 2021: Multimodal Emotion Recognition on Comics scenes (EmoRecCom)")

parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help="path to the download train data directory",
)

parser.add_argument(
    "--target_cols",
    nargs="+",
    default=ALL_COLS,
    help="define columns for forecasting",
)

parser.add_argument(
    "--gpus",
    nargs="+",
    default=0,
    help="select gpus to use",
)

parser.add_argument(
    "--do_train",
    action="store_true",
    default=False,
    help="whether train the pretrained model with provided train data",
)

parser.add_argument(
    "--do_infer",
    action="store_true",
    default=False,
    help="whether predict the provided test data with the trained models from checkpoint directory",
)

parser.add_argument(
    "--ckpt_dir",
    type=str,
    # default="outputs/efn-b0_64_roberta-base_32_-1",
    help="path to the directory containing checkpoints (.h5) models",
)

parser.add_argument(
    "--image_model",
    default="efn-b0",
    type=str,
    help=f"pretrained image model name in list \n {IMAGE_MODELS} \n None for using unimodal model",
)

parser.add_argument(
    "--bert_model",
    default="roberta-base",
    type=str,
    help="path to pretrained bert model path or directory (e.g: https://huggingface.co/models)",
)

parser.add_argument(
    "--word_embedding",
    type=str,
    help=f"path to a pretrained static word embedding in list \n {WORD_EMBEDDING_MODELS} \n None for using bert model to represent text",
)

parser.add_argument(
    "--max_vocab",
    # default=30000,
    type=int,
    help="maximum of word in the vocabulary (Tensorflow word tokenizer)",
)

parser.add_argument(
    "--max_word",
    default=-1,
    type=int,
    help="maximum word per text sample (Tensorflow word tokenizer)",
)

parser.add_argument(
    "--image_size",
    # default=256,
    type=int,
    help="size of image",
)

parser.add_argument(
    "--max_len",
    default=64,
    type=int,
    help="max sequence length for padding and truncation (Bert word tokenizer)",
)

parser.add_argument(
    "--lower",
    action="store_true",
    help="whether lowercase text or not",
)

parser.add_argument(
    "--text_separator",
    default=" ",
    type=str,
    help="define separator to join conversations",
)

parser.add_argument(
    "--n_hiddens",
    default=-1,
    type=int,
    help="concatenate n_hiddens final layer to get sequence's bert embedding, -1 for using [CLS] token embedding only",
)

parser.add_argument(
    "--lr",
    default=3e-5,
    type=float,
    help="Learning rate",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--n_epochs",
    default=5,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--kaggle",
    action="store_true",
    help="whether using kaggle environment or not",
)

parser.add_argument(
    "--seed",
    default=1710,
    type=int,
    help="seed for reproceduce",
)

args = parser.parse_args()


def extract_fold_number(file_path: str) -> str:
    return int(re.findall("(\d)\.h5", file_path)[0])


if __name__ == "__main__":
    seed_all(seed=args.seed)
    print_signature()

    # setup working directory
    experiment = f"{args.image_model}_{args.image_size}_{args.bert_model}_{args.max_len}_{args.n_hiddens}"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = custom_logger(logging_dir=OUTPUT_DIR)

    config_info = "\n" + "*" * 50 + "\nGLOBAL CONFIGURATION\n"
    for arg in vars(args):
        config_info += f"{arg} : { getattr(args, arg)}\n"
    config_info += "*" * 50
    logger.info(config_info)

    with open(f"{OUTPUT_DIR}/config.yaml", "w") as file:
        yaml.dump(vars(args), file, indent=4)

    if "KAGGLE_CONTAINER_NAME" in os.environ:
        KAGGLE = True
        from kaggle_datasets import KaggleDatasets

        GCS_DS_PATH = KaggleDatasets().get_gcs_path()
    else:
        logger.warning("Kaggle enviroment is not available, use local storage instead")

    # Adapt working directory
    TRAIN_IMAGE_DIR = os.path.join(args.data_dir, TRAIN_IMAGE_DIR)
    TEST_IMAGE_DIR = os.path.join(args.data_dir, TEST_IMAGE_DIR)
    TRAIN_LABELS = os.path.join(args.data_dir, TRAIN_LABELS)
    TRAIN_POLARITY = os.path.join(args.data_dir, TRAIN_POLARITY)
    TRAIN_SCRIPT = os.path.join(args.data_dir, TRAIN_SCRIPT)
    TEST_SCRIPT = os.path.join(args.data_dir, TEST_SCRIPT)
    SAMPLE_SUBMISSION = os.path.join(args.data_dir, SAMPLE_SUBMISSION)
    TARGET_COLS = args.target_cols
    TARGET_SIZE = len(TARGET_COLS)

    GPUS = ",".join([str(gpu) for gpu in args.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

    train_polarity = pd.read_csv(TRAIN_POLARITY, index_col=0)
    train_labels = pd.read_csv(TRAIN_LABELS, index_col=0)
    with open(TRAIN_SCRIPT, "r") as f:
        train_script_json = json.load(f)

    sample_submission = pd.read_csv(SAMPLE_SUBMISSION, index_col=0, header=None, names=["image_id"] + ALL_COLS)
    with open(TEST_SCRIPT, "r") as f:
        test_script_json = json.load(f)

    train_script = pd.DataFrame(train_script_json)
    test_script = pd.DataFrame(test_script_json)

    train_script = train_script.rename(columns={"img_id": "image_id"})
    test_script = test_script.rename(columns={"img_id": "image_id"})

    train_no_script = pd.merge(train_labels, train_polarity, on="image_id")
    train_non_processed = pd.merge(train_no_script, train_script, on="image_id")
    test_non_processed = pd.merge(sample_submission, test_script, on="image_id")

    train_processed = process_dialog(
        process_emotion_polarity(
            add_file_path(df=train_non_processed, image_dir=TRAIN_IMAGE_DIR, gcs_ds_path=GCS_DS_PATH)
        ),
        lower=args.lower,
        text_separator=args.text_separator,
    )
    test_processed = process_dialog(
        add_file_path(df=test_non_processed, image_dir=TEST_IMAGE_DIR, gcs_ds_path=GCS_DS_PATH),
        lower=args.lower,
        text_separator=args.text_separator,
    )

    # for data analysis purpose
    with open("train.txt", "w") as f:
        for index, row in train_processed.iterrows():
            f.write(f"\n{index} {row['image_id']} \n")
            for col in ALL_COLS:
                f.write(f"{col} {row[col]} | ")
            f.write(f"\nText: {row['text']}\n")
            f.write(f"Narration {row['narration']}\n")

    with open("test.txt", "w") as f:
        for index, row in test_processed.iterrows():
            f.write(f"\n{index} {row['image_id']} \n")
            for col in ALL_COLS:
                f.write(f"{col} {row[col]} | ")
            f.write(f"\nText: {row['text']}\n")
            f.write(f"Narration {row['narration']}\n")

    # Get all text
    train_texts = train_processed["text"].tolist()
    test_texts = test_processed["text"].tolist()
    all_texts = train_texts + test_texts

    strategy = select_strategy()

    if args.do_train:
        logger.info("Start training")
        WORD_EMBEDDING_MODEL = (
            os.path.splitext(os.path.basename(args.word_embedding))[0] if args.word_embedding else None
        )
        MULTIMODAL = args.image_model in IMAGE_MODELS
        STATIC_WORD_EMBEDDING = WORD_EMBEDDING_MODEL in WORD_EMBEDDING_MODELS

        if MULTIMODAL:
            logger.info("Training with multi-modal strategy")
        else:
            args.image_size = -1
            logger.warning("Training without multi-modal strategy")
        if STATIC_WORD_EMBEDDING:
            logger.info("Training with additional word embedding features")
        else:
            logger.warning("Training without additional word embedding features")

        if not MULTIMODAL and args.image_model:
            raise NotImplementedError(
                f"{args.image_model} is not available, please choose one of the following model\n {IMAGE_MODELS}"
            )
        if not STATIC_WORD_EMBEDDING and args.word_embedding:
            raise NotImplementedError(
                f"{args.word_embedding} is not available, please choose one of the following model\n {STATIC_WORD_EMBEDDING}"
            )

        embedding_matrix = None
        tf_tokenizer = None
        if STATIC_WORD_EMBEDDING:
            tf_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=args.max_vocab, lower=args.lower)
            tf_tokenizer.fit_on_texts(all_texts)
            logger.info(f"Word vocabulary size: {len(tf_tokenizer.word_index)}")
            tf_tokenizer_fp = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
            pickle.dump(tf_tokenizer, open(tf_tokenizer_fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved TF tokenizer to {tf_tokenizer_fp}")
            embedding_matrix = get_embedding_matrix(
                word_index=tf_tokenizer.word_index,
                max_vocab=args.max_vocab,
                embedding_fp=args.word_embedding,
                lower=args.lower,
            )

        if args.do_infer:
            args.ckpt_dir = OUTPUT_DIR

        model = build_model(
            image_model=args.image_model,
            bert_model=args.bert_model,
            image_size=args.image_size,
            max_len=args.max_len,
            max_word=args.max_word,
            embedding_matrix=embedding_matrix,
            target_size=TARGET_SIZE,
            n_hiddens=args.n_hiddens,
        )
        model.summary(print_fn=logger.info)
        tf.keras.utils.plot_model(model, to_file=OUTPUT_DIR + "/model.png")
        folds_history = []
        folds = train_processed["fold"].unique()
        for fold in sorted(folds):
            logger.info("*" * 100)
            logger.info(f"FOLD: {fold+1}/{len(folds)}")
            K.clear_session()
            #     with strategy.scope():
            #         model = build_model()
            model = build_model(
                image_model=args.image_model,
                bert_model=args.bert_model,
                image_size=args.image_size,
                max_len=args.max_len,
                max_word=args.max_word,
                embedding_matrix=embedding_matrix,
                target_size=TARGET_SIZE,
                n_hiddens=args.n_hiddens,
            )
            model.compile(
                tf.keras.optimizers.Adam(lr=args.lr),
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(multi_label=TARGET_SIZE > 1)],
            )
            reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

            model_dir = os.path.join(OUTPUT_DIR, f"Fold_{fold}.h5")

            sv = tf.keras.callbacks.ModelCheckpoint(
                model_dir,
                monitor="val_auc",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                save_freq="epoch",
            )

            train_df = train_processed[train_processed["fold"] != fold].copy()
            val_df = train_processed[train_processed["fold"] == fold].copy()
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)
            train_dataset = ICDARGenerator(
                df=train_df,
                bert_tokenizer=bert_tokenizer,
                tf_tokenizer=tf_tokenizer,
                shuffle=True,
                batch_size=args.batch_size,
                image_size=args.image_size,
                target_cols=TARGET_COLS,
                max_len=args.max_len,
                max_word=args.max_word,
            )

            val_dataset = ICDARGenerator(
                df=val_df,
                bert_tokenizer=bert_tokenizer,
                tf_tokenizer=tf_tokenizer,
                shuffle=True,
                batch_size=args.batch_size,
                image_size=args.image_size,
                target_cols=TARGET_COLS,
                max_len=args.max_len,
                max_word=args.max_word,
            )

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                callbacks=[
                    sv,
                    # reduce_lr,
                ],
                epochs=args.n_epochs,
                #         verbose = 1
            )
            for k, v in history.history.items():
                logger.info(f"{k} : {v}")
            folds_history.append(history.history)

    if args.do_infer:

        if not args.do_train:
            logger.info(f"Inferencing with models from {args.ckpt_dir}")
            logger.warning("The initial passing arguments with be overwriten with configuration from this checkpoint")

        with open(f"{args.ckpt_dir}/config.yaml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in config.items():
                if key in LOAD_ARGS:
                    setattr(args, key, value)

        config_info = "\n" + "*" * 50 + "\nGLOBAL CONFIGURATION\n"

        for arg in vars(args):
            config_info += f"{arg} : { getattr(args, arg)}\n"
        config_info += "*" * 50
        logger.info(config_info)

        logger.info("Start inferencing")

        embedding_matrix = None
        if args.word_embedding:
            tf_tokenizer_fp = os.path.join(args.ckpt_dir, "tokenizer.pkl")
            tf_tokenizer = pickle.load(open(tf_tokenizer_fp, "rb"))
            embedding_matrix = get_embedding_matrix(
                word_index=tf_tokenizer.word_index,
                max_vocab=args.max_vocab,
                embedding_fp=args.word_embedding,
                lower=args.lower,
            )

        model = build_model(
            image_model=args.image_model,
            bert_model=args.bert_model,
            image_size=args.image_size,
            max_len=args.max_len,
            max_word=args.max_word,
            embedding_matrix=embedding_matrix,
            target_size=len(args.target_cols),
            n_hiddens=args.n_hiddens,
        )
        model.summary(print_fn=logger.info)
        preds = []
        oof_preds = []

        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)
        test_dataset = ICDARGenerator(
            test_processed,
            bert_tokenizer=bert_tokenizer,
            tf_tokenizer=tf_tokenizer,
            shuffle=False,
            batch_size=1,
            image_size=args.image_size,
            target_cols=args.target_cols,
            max_len=args.max_len,
            max_word=args.max_word,
        )
        for i, file_path in enumerate(glob(f"{args.ckpt_dir}/*.h5")):
            K.clear_session()
            logger.info(f"Inferencing with model from: {file_path}")
            fold = extract_fold_number(file_path)
            val_df = train_processed[train_processed["fold"] == fold].copy()
            val_dataset = ICDARGenerator(
                df=val_df,
                bert_tokenizer=bert_tokenizer,
                tf_tokenizer=tf_tokenizer,
                shuffle=False,
                batch_size=1,
                image_size=args.image_size,
                target_cols=args.target_cols,
                max_len=args.max_len,
                max_word=args.max_word,
            )
            model.load_weights(file_path)
            pred = model.predict(test_dataset, verbose=1)
            oof_pred = model.predict(val_dataset, verbose=1)
            preds.append(pred)
            val_df[TARGET_COLS] = oof_pred
            oof_preds.append(val_df)
        pred = np.mean(preds, axis=0)
        test_processed[TARGET_COLS] = pred
        result_path = os.path.join(OUTPUT_DIR, "results.csv")
        test_processed[["image_id"] + TARGET_COLS].to_csv(result_path, header=False)
        test_path = os.path.join(OUTPUT_DIR, "test_pred.npy")
        with open(test_path, "wb") as f:
            np.save(f, test_processed[TARGET_COLS].values)
        oof_df = pd.DataFrame.sort_index(pd.concat(oof_preds, axis=0))
        oof_path = os.path.join(OUTPUT_DIR, "oof_pred.npy")
        with open(oof_path, "wb") as f:
            np.save(f, oof_df[TARGET_COLS].values)
