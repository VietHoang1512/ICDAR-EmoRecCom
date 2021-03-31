<div align="center">

# [ICDAR 2021](https://icdar2021.org/program-2/competitions/) <img src="assets/icdar.png" alt="ICDAR 2021" width="25" height="15">

## [Multimodal Emotion Recognition on Comics scenes](https://competitions.codalab.org/competitions/27884)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1022cd5ee1a34cb8bea336adef0d7c26)](https://www.codacy.com/gh/VietHoang1710/EmoRecCom/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=VietHoang1710/EmoRecCom&amp;utm_campaign=Badge_Grade)

</div>

## Overview
- This is the code for my submission of to EmoRecCom leaderboard. It is implemented under Tensorflow 2.0 framework
- For usage of this code, please follow [here](src/README.md)
- The ensemble models (TF + Pytorch) achieved 0.685 in the private leaderboard

## Data preparation 

### Competition data
- The data folder is organized as presented in [here](src/utils/constant.py), you can also edit this file to adapt to your working directory (not recommended). Instead, this could be directly downloaded from drive by running `setup.sh`

- The data directory by default is as follow:
```
├── private_test
│   ├── images
│   ├── readme.md
│   ├── results.csv
│   └── transcriptions.json
├── public_test
│   ├── images
│   ├── results.csv
│   └── transcriptions.json
└── public_train
    ├── additional_infor:train_emotion_polarity.csv
    ├── images
    ├── readme.md
    ├── train_5_folds.csv
    ├── train_emotion_labels.csv
    └── train_transcriptions.json
```
### Additional data (optional)
- In case you want to train a model with static word embeddings (word2vec, glove, fasttext, etc.). Download them by uncommenting the desired pretrained models you want in `setup.sh`. By default, static word embedding is not used in our approach
- The provided static embedding models are in pickle file for easy loading, refer `prepare_data.sh` for more detail

## Prerequisites
- tensorflow
- numpy
- pandas
- sklearn
- transformers
- efficientnet

Running `setup.sh` also installs the dependencies
## Train & inference
- Example bash scripts for training and inference are `train.sh` and `infer.sh`
- In addition, we perform [stacking](src/scripts/stacking.py) by Logistic Regression, require out-of-fold along with test prediction
## Outputs
- Folder contains all TF [experiments](https://drive.google.com/drive/folders/1mfeWRV9-yfmcbIWgLWLBKblM1-cPaWOi?usp=sharing)
## Reproducing:
- Best single model (0.676 ROC-AUC) [configuration](assets/config.yaml)