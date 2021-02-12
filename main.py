import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from tensorflow.keras import backend as K
from transformers import AutoTokenizer

from trainer.dataset import ICDARGenerator
from trainer.model import build_model
from trainer.utils import scheduler, seed_all, select_strategy
from utils.constant import *
from utils.logger import custom_logger
from utils.processor import (
    add_file_path,
    process_dialog,
    process_emotion_polarity,
)
from utils.signature import print_signature

print("Using Tensorflow version:", tf.__version__)
print("Using Transformers version:", transformers.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()

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
    "--do_train",
    default=True,
    type=bool,
    help="whether train the pretrained model with provided train data",
)

parser.add_argument(
    "--do_infer",
    default=True,
    type=bool,
    help="whether predict the pretrained model with provided test data",
)

parser.add_argument(
    "--img_model",
    default="efn_b0",
    type=str,
    help="pretrained image model name",
)

parser.add_argument(
    "--bert_model",
    default="roberta-base",
    type=str,
    help="path to pretrained bert model path or directory",
)

parser.add_argument(
    "--img_size",
    default=256,
    type=int,
    help="size of image",
)

parser.add_argument(
    "--max_len",
    default=64,
    type=int,
    help="max sequence length for padding and truncation",
)

parser.add_argument(
    "--n_hiddens",
    default=-1,
    type=int,
    help="concatenate n_hiddens final layer to get sequence's bert embedding, -1 for using [CLS] token embedding only",
)

parser.add_argument(
    "--batch_size",
    default=4,
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
    default=False,
    type=bool,
    help="whether using kaggle environment or not",
)

parser.add_argument(
    "--seed",
    default=1710,
    type=int,
    help="seed for reproceduce",
)

args = parser.parse_args()

if __name__ == "__main__":
    seed_all(seed=args.seed)
    print_signature()

    # setup working directory
    experiment = f"{args.img_model}_{args.bert_model}_{args.max_len}"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = custom_logger(logging_dir=OUTPUT_DIR)

    config_info = "\n" + "*" * 50 + "\nGLOBAL CONFIGURATION\n"
    for arg in vars(args):
        config_info += f"{arg} : { getattr(args, arg)}\n"
    config_info += "*" * 50
    logger.info(config_info)

    if "KAGGLE_CONTAINER_NAME" in os.environ:
        KAGGLE = True
        from kaggle_datasets import KaggleDatasets

        GCS_DS_PATH = KaggleDatasets().get_gcs_path()
    else:
        logger.warning("Kaggle enviroment is not available, use local storage instead")

    # Adapt working directory
    TRAIN_IMG_DIR = os.path.join(args.data_dir, TRAIN_IMG_DIR)
    TEST_IMG_DIR = os.path.join(args.data_dir, TEST_IMG_DIR)
    TRAIN_LABELS = os.path.join(args.data_dir, TRAIN_LABELS)
    TRAIN_POLARITY = os.path.join(args.data_dir, TRAIN_POLARITY)
    TRAIN_SCRIPT = os.path.join(args.data_dir, TRAIN_SCRIPT)
    TEST_SCRIPT = os.path.join(args.data_dir, TEST_SCRIPT)
    SAMPLE_SUBMISSION = os.path.join(args.data_dir, SAMPLE_SUBMISSION)
    TARGET_COLS = args.target_cols
    TARGET_SIZE = len(TARGET_COLS)
    MULTIMODAL = args.img_model in IMAGE_MODELS

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
        process_emotion_polarity(add_file_path(df=train_non_processed, img_dir=TRAIN_IMG_DIR, gcs_ds_path=GCS_DS_PATH))
    )
    test_processed = process_dialog(add_file_path(df=test_non_processed, img_dir=TEST_IMG_DIR, gcs_ds_path=GCS_DS_PATH))

    # for data analysis purpose
    with open("train.txt", "w") as f:
        f.write("\n".join(train_processed["text"]))
    with open("test.txt", "w") as f:
        f.write("\n".join(test_processed["text"]))

    strategy = select_strategy()

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)

    if args.do_train:
        logger.info("Start training")
        model = build_model(
            img_model=args.img_model,
            bert_model=args.bert_model,
            image_size=args.img_size,
            max_len=args.max_len,
            target_size=TARGET_SIZE,
            n_hiddens=args.n_hiddens,
        )
        model.summary(print_fn=logger.info)
        tf.keras.utils.plot_model(model, to_file=OUTPUT_DIR + "/model.png")
        folds_history = []
        folds = train_processed["fold"].unique()
        for fold in sorted(folds):
            print("*" * 100)
            print(f"FOLD: {fold+1}/{len(folds)}")
            K.clear_session()
            #     with strategy.scope():
            #         model = build_model()
            model = build_model(
                img_model=args.img_model,
                bert_model=args.bert_model,
                image_size=args.img_size,
                max_len=args.max_len,
                target_size=TARGET_SIZE,
                n_hiddens=args.n_hiddens,
            )
            reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

            model_dir = os.path.join(OUTPUT_DIR, f"Fold_{fold+1}.h5")

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

            train_dataset = ICDARGenerator(
                df=train_df,
                multimodal=MULTIMODAL,
                bert_tokenizer=bert_tokenizer,
                shuffle=True,
                batch_size=args.batch_size,
                image_size=args.img_size,
                target_cols=TARGET_COLS,
                max_len=args.max_len,
            )

            val_dataset = ICDARGenerator(
                df=val_df,
                multimodal=MULTIMODAL,
                bert_tokenizer=bert_tokenizer,
                shuffle=True,
                batch_size=args.batch_size,
                image_size=args.img_size,
                target_cols=TARGET_COLS,
                max_len=args.max_len,
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
            folds_history.append(history.history)
    if args.do_infer:
        logger.info("Start inference")
        test_dataset = ICDARGenerator(
            test_processed,
            multimodal=MULTIMODAL,
            bert_tokenizer=bert_tokenizer,
            shuffle=False,
            batch_size=1,
            image_size=args.img_size,
            target_cols=TARGET_COLS,
            max_len=args.max_len,
        )

        preds = []
        for i, file_name in enumerate(glob(f"{OUTPUT_DIR}/*.h5")):
            print("*" * 100)
            K.clear_session()
            model_path = os.path.join(OUTPUT_DIR, file_name)

            print(f"Inferencing with model from: {model_path}")
            model.load_weights(model_path)

            pred = model.predict(test_dataset, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0)
        test_processed[TARGET_COLS] = pred
        result_path = os.path.join(OUTPUT_DIR, "results.csv")
        test_processed[["image_id"] + TARGET_COLS].to_csv(result_path, header=False)
