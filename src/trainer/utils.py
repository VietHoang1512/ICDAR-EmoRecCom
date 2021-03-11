import os
import random
from typing import Dict

import numpy as np
import tensorflow as tf


def parse_experiment(experiment: str) -> Dict:
    """
    Parse experiment name into kwargs for alignment between train-infer

    Args:
        experiment (str): Name of the experiment directory

    Returns:
        Dict: keyword arguments to build a ICDAR model
    """
    fields = ["image_model", "image_size", "bert_model", "max_len", "n_hiddens"]
    values = experiment.split("_")

    return dict(zip(fields, values))


def select_strategy():
    """
    Auto select device to run in (TPU or GPU)
    Restrict reserving all GPU RAM in tensorflow

    Returns:
        stragegy: strategy for training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
            print("Running on GPU:", gpus)
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def seed_all(seed=1512):
    """
    Set seed for reproducing result
    Args:
        seed (int, optional): seed number. Defaults to 1512.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)


def scheduler(epoch: int) -> float:
    """
    Learning rate scheduler by epoch
    Args:
        epoch (int): Epoch number
    Returns:
        float: learning rate in epochs
    """
    return 3e-5 * 0.2 ** epoch


if __name__ == "__main__":
    print(parse_experiment("efn-b1_128_roberta-base_48_-1"))
