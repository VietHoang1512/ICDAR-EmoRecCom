import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from utils.constant import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

STACKING_DIR = "data/stacking"
N_FOLDS = 5
oof_df = pd.read_csv("data/train_5_folds.csv", index_col=0)
test_df = pd.read_csv("data/results.csv", index_col=0, header=None, names=["image_id"] + ALL_COLS)

oof_pred_dfs = []
test_pred_dfs = []

EXPERIMENTS = [
    "efn_b5_128_roberta-base_48_2",
    "efn_b5_128_bert-base-cased_48",
    "efn_b5_128_roberta-base_48",
    "efn_b5_128_distilbert-base-uncased_48",
    "efn_b5_128_roberta-base_64",
    "efn_b5_128_bert-base-uncased_48",
]

for exp in EXPERIMENTS:
    oof_pred_fp = os.path.join(STACKING_DIR, exp, "oof_pred.npy")
    test_pred_fp = os.path.join(STACKING_DIR, exp, "test_pred.npy")
    oof_pred = np.load(oof_pred_fp)
    test_pred = np.load(test_pred_fp)
    oof_pred_df = oof_df.copy()[ALL_COLS]
    test_pred_df = test_df.copy()[ALL_COLS]
    oof_pred_df[ALL_COLS] = oof_pred
    test_pred_df[ALL_COLS] = test_pred
    for col in ALL_COLS:
        oof_pred_df = oof_pred_df.rename(columns={col: f"{exp}_{col}"})
        test_pred_df = test_pred_df.rename(columns={col: f"{exp}_{col}"})
    oof_pred_dfs.append(oof_pred_df)
    test_pred_dfs.append(test_pred_df)
oof_pred_df = pd.concat(oof_pred_dfs, axis=1)
test_pred_df = pd.concat(test_pred_dfs, axis=1)


def extract_column(column_name):
    return column_name.split("_")[-1]


params_cat = {
    "random_seed": 1710,
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 10,
    "thread_count": 4,
    "od_type": "Iter",
    "od_wait": 20,
    "task_type": "CPU",
    "eval_metric": "AUC",
    "use_best_model": True,
}

params_lgb = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 16,
    "max_bin": 256,
    "verbosity": 0,
    "force_col_wise": True,
}

multi_auc_scores = []
for target_col in ALL_COLS:
    print("STACKING ON COLUMNS:", target_col)
    oof_pred_df_single = oof_pred_df.copy()[[col for col in oof_pred_df.columns if extract_column(col) == target_col]]
    test_pred_df_single = test_pred_df.copy()[
        [col for col in test_pred_df.columns if extract_column(col) == target_col]
    ]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    test_preds = []
    oof_scores = []
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(oof_df)):
        X_train = oof_pred_df_single.iloc[train_idx]
        y_train = oof_df[target_col].iloc[train_idx]
        X_val = oof_pred_df_single.iloc[val_idx]
        y_val = oof_df[target_col].iloc[val_idx]

        # """
        #     CATBOOST
        # """
        # dtrain = Pool(X_train, label=y_train)
        # dval = Pool(X_val, label=y_val)
        # clf = CatBoostClassifier(**params_cat)
        # clf.fit(dtrain, eval_set = dval, verbose = 0, early_stopping_rounds = 500)
        # val_pred = clf.predict_proba(X_val)[:, 1]
        # test_pred = clf.predict_proba(test_pred_df_single)[:, 1]

        """
            LGB
        """
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        clf = lgb.train(
            params_lgb,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            # feval = evalerror,
            verbose_eval=False,
            early_stopping_rounds=200,
        )
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(test_pred_df_single)

        test_preds.append(test_pred)
        oof_score = roc_auc_score(y_val, val_pred)
        oof_scores.append(oof_score)
        print(f"FOLD {fold_id} {oof_score}")
    target_column_score = np.mean(oof_scores)
    print("TARGET COLUMN SCORE: ", target_column_score)
    multi_auc_scores.append(target_column_score)
    test_pred = np.mean(test_preds, axis=0)
    test_df[target_col] = test_pred
print("OVERALL SCORE", np.mean(multi_auc_scores))
test_df[["image_id"] + ALL_COLS].to_csv("results.csv", header=False)
