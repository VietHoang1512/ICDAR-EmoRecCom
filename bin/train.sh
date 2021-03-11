#!/bin

export PYTHONPATH=$PWD

data_dir="data"
embedding_dir="embeddings/"

# word_embedding="$embedding_dir/wiki.en.pkl"
# word_embedding="$embedding_dir/wiki-news-300d-1M.pkl"
# word_embedding="$embedding_dir/crawl-300d-2M.pkl"
word_embedding="$embedding_dir/glove.840B.300d.pkl"

python src/main.py \
    --data_dir=$data_dir \
    --target_cols angry disgust fear happy sad surprise neutral other \
    --gpus 1 \
    --image_model efn-b1 \
    --bert_model roberta-base \
    --word_embedding=$word_embedding \
    --max_vocab 30000 \
    --image_size 128 \
    --max_word 36 \
    --max_len 48 \
    --lower \
    --text_separator " " \
    --n_hiddens -1 \
    --lr 0.00003 \
    --n_epochs 5 \
    --seed 1710 \
    --do_train 