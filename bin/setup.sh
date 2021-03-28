#!/bin/bash

# pip install -r requirements.txt


data_dir="data"
emb_dir="embeddings/"

[[ -d $data_dir ]] || ((gdown --id 1bofnf_jiELtKuSQfU8lXy6zDKutWUgzs) && (unzip data.zip ))
[[ ! -f data.zip ]] || (rm data.zip)

mkdir -p $emb_dir

[[ -f "$emb_dir/glove.840B.300d.pkl" ]] || (gdown --id 1-7SQ49tYL2STnSkVrgwXwd_wyKkt5oMF -O $emb_dir)

# [[ -f "$emb_dir/wiki.en.vec.pkl" ]] || (gdown --id 1-XwvKbEpZuIiLRK9O0Oa8OO8BnaC7LS9 -O $emb_dir)

# [[ -f "$emb_dir/crawl-300d-2M.vec.pkl" ]] || (gdown --id 1-Gk8uChzM5Iym9GC_LifzrQCMhodIJXW -O $emb_dir)

# [[ -f "$emb_dir/wiki-news-300d-1M.vec.pkl" ]] || (gdown --id 1-GhBuFTuTOsX1oreXRhSFF4L9Za4Xl7K -O $emb_dir)