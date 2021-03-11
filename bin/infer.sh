#!/bin

export PYTHONPATH=$PWD

data_dir="data"
embedding_dir="embeddings/"
ckpt_dir="outputs/efn-b1_128_roberta-base_48_-1"

# word_embedding="$embedding_dir/wiki.en.pkl"
# word_embedding="$embedding_dir/wiki-news-300d-1M.pkl"
# word_embedding="$embedding_dir/crawl-300d-2M.pkl"
# word_embedding="$embedding_dir/glove.840B.300d.pkl"

python src/main.py \
    --data_dir=$data_dir \
    --target_cols angry disgust fear happy sad surprise neutral other \
    --gpus 1 \
    --ckpt_dir $ckpt_dir \
    --do_infer \