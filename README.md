<div align="center">

# [ICDAR 2021](https://icdar2021.org/program-2/competitions/) <img src="assets/icdar.png" alt="ICDAR 2021" width="25" height="15">

## [Multimodal Emotion Recognition on Comics scenes](https://competitions.codalab.org/competitions/27884)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4b2bb47d1499410cad4f148c4c09dbe4)](https://www.codacy.com/gh/VietHoang1512/ICDAR-EmoRecCom/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=VietHoang1512/ICDAR-EmoRecCom&amp;utm_campaign=Badge_Grade)

</div>

## Overview
- This is the code for EmoRecCom #1 solution (Tensorflow 2.0 version) 
- For usage of this code, please follow [here](src/README.md)
- The ensemble models (TF + Pytorch) achieved 0.685 in the private leaderboard [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_51)

<div align="center">
<img src="assets/leaderboard.png" alt="Track 4 private leader board" width="600" height="240">
</div>

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
- In case you want to train a model with static word embeddings (word2vec, glove, fasttext, etc.). Download them by uncommenting the desired pretrained models in `setup.sh`. By default, static word embedding is not used in our approach
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
### Train example
```sh
python src/main.py \
    --train_dir data/public_train \
    --target_cols angry disgust fear happy sad surprise neutral other \
    --gpus 0 1 2 \
    --image_model efn-b2 \
    --bert_model roberta-base \
    --word_embedding embeddings/glove.840B.300d.pkl \
    --max_vocab 30000 \
    --image_size 256 \
    --max_word 36 \
    --max_len 48 \
    --text_separator " " \
    --n_hiddens -1 \
    --lr 0.00003 \
    --n_epochs 5 \
    --seed 1710 \
    --do_train \
    --lower \
```

### Inference example

```sh
python src/main.py \
    --test_dir data/private_test \
    --target_cols angry disgust fear happy sad surprise neutral other \
    --gpus 1 \
    --ckpt_dir outputs/efn-b2_256_roberta-base_48_-1_0.1/ \
    --do_infer \
```
- In addition, we perform [stacking](src/scripts/stacking.py) by Logistic Regression, requires out-of-fold along with test prediction

## Outputs
- Folder containing all TF [experiments](https://drive.google.com/drive/folders/1mfeWRV9-yfmcbIWgLWLBKblM1-cPaWOi?usp=sharing)

## Reproducing:
<div align="center">
<br>
<img src="assets/model.png" alt="Model achitecture" width="700" height="290">
<br>
</div>

- Technical report: [English](https://docs.google.com/presentation/d/1ioExeoDKOnT2KIPHeY3bt3Hfj5AOPdu2AXejgbTtEoE/edit?usp=sharing) / [Vietnamese](assets/ML_Report.pdf)
  
- Best single model (0.676 ROC-AUC) [configuration](assets/config.yaml)
