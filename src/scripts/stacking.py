import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.utils import constant

STACKING_DIR = "outputs"
N_FOLDS = 5
SEED = 1
oof_df = pd.read_csv("data/public_train/train_5_folds.csv", index_col=0)
test_df = pd.read_csv(
    "data/private_test/results.csv",
    index_col=0,
    header=None,
    names=["image_id"] + constant.ALL_COLS,
)

oof_pred_dfs = []
test_pred_dfs = []

EXPERIMENTS = os.listdir(STACKING_DIR)

for exp in EXPERIMENTS:
    oof_pred_fp = os.path.join(STACKING_DIR, exp, "oof_pred.npy")
    test_pred_fp = os.path.join(STACKING_DIR, exp, "test_pred.npy")
    oof_pred = np.load(oof_pred_fp)
    test_pred = np.load(test_pred_fp)
    oof_pred_df = oof_df.copy()[constant.ALL_COLS]
    test_pred_df = test_df.copy()[constant.ALL_COLS]
    oof_pred_df[constant.ALL_COLS] = oof_pred
    test_pred_df[constant.ALL_COLS] = test_pred
    for col in constant.ALL_COLS:
        oof_pred_df = oof_pred_df.rename(columns={col: f"{exp}_{col}"})
        test_pred_df = test_pred_df.rename(columns={col: f"{exp}_{col}"})
    oof_pred_dfs.append(oof_pred_df)
    test_pred_dfs.append(test_pred_df)
oof_pred_df = pd.concat(oof_pred_dfs, axis=1)
test_pred_df = pd.concat(test_pred_dfs, axis=1)


def extract_column(column_name):
    return column_name.split("_")[-1]


multi_auc_scores = []
for target_col in constant.ALL_COLS:
    print("PREDICTING COLUMN:", target_col)
    oof_pred_df_single = oof_pred_df.copy()[[col for col in oof_pred_df.columns if extract_column(col) == target_col]]
    test_pred_df_single = test_pred_df.copy()[
        [col for col in test_pred_df.columns if extract_column(col) == target_col]
    ]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = []
    oof_scores = []
    for fold_id in range(N_FOLDS):
        train_idx = oof_df[oof_df["fold"] != fold_id].index.tolist()
        val_idx = oof_df[oof_df["fold"] == fold_id].index.tolist()
        X_train = oof_pred_df_single.iloc[train_idx]
        y_train = oof_df[target_col].iloc[train_idx]
        X_val = oof_pred_df_single.iloc[val_idx]
        y_val = oof_df[target_col].iloc[val_idx]
        clf = LogisticRegression(C=0.005, n_jobs=4, penalty="l2")
        # clf = SVC(probability=True)
        clf.fit(X_train, y_train)

        val_pred = clf.predict_proba(X_val)[:, 1]
        test_pred = clf.predict_proba(test_pred_df_single)[:, 1]
        test_preds.append(test_pred)
        oof_score = roc_auc_score(y_val, val_pred)
        oof_scores.append(oof_score)
        # print(f"FOLD {fold_id} {oof_score}")
    target_column_score = np.mean(oof_scores)
    print("TARGET COLUMN SCORE: ", target_column_score)
    multi_auc_scores.append(target_column_score)
    test_pred = np.mean(test_preds, axis=0)
    test_df[target_col] = test_pred
print("OVERALL SCORE", np.mean(multi_auc_scores))
test_df[["image_id"] + constant.ALL_COLS].to_csv("results.csv", header=False)
