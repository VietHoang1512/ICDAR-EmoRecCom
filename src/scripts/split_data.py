"""
    K-fold split data for comparation
"""

import pandas as pd
from sklearn.model_selection import KFold


def kfold_split(train_df, n_folds):
    """
    K-fold split data for further comparation

    Args:
        train_df (DataFrame): train dataframe
        n_folds (int): number of fold

    Returns:
        DataFrame: data with fold idices
    """
    train_df = train_df.copy()
    train_df["fold"] = 0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, "fold"] = fold_id
    return train_df


if __name__ == "__main__":
    n_folds = 5
    train_df = pd.read_csv("data/train_emotion_labels.csv")
    train_df = kfold_split(train_df, n_folds=n_folds)
    train_df.to_csv(f"data/train_{n_folds}_folds.csv", index=False)
    print("DONE!")
