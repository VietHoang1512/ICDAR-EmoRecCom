import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


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


def length_plot(lengths):
    """
    Plot the sequence length statistic
    Args:
        lengths (list): Sequence lengths (by word or character)
    """
    plt.figure(figsize=(15, 9))
    textstr = f" Mean: {np.mean(lengths):.2f} \u00B1 {np.std(lengths):.2f} \n Max: {np.max(lengths)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    plt.text(0, 0, textstr, fontsize=14, verticalalignment="top", bbox=props)
    sns.countplot(lengths, orient="h")
    plt.show()


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


def scheduler(epoch):
    """
    Learning rate scheduler by epoch
    Args:
        epoch (int): Epoch number
    Returns:
        float: learning rate in epochs
    """
    return 3e-5 * 0.2 ** epoch
