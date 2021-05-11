#!/bin/bash
export PYTHONPATH=$PWD

emb_dir="embeddings/"
mkdir -p $emb_dir

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip -d $emb_dir
rm crawl-300d-2M.vec.zip
python scripts/prepare_embedding.py \
    -i "$emb_dir/crawl-300d-2M.vec"
rm "$emb_dir/crawl-300d-2M.vec"

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip -d $emb_dir
rm wiki-news-300d-1M.vec.zip
python scripts/prepare_embedding.py \
    -i "$emb_dir/wiki-news-300d-1M.vec"
rm "$emb_dir/wiki-news-300d-1M.vec"

wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec -P $emb_dir
python scripts/prepare_embedding.py \
    -i "$emb_dir/wiki.en.vec"
rm $emb_dir/wiki.en.vec

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d $emb_dir
rm glove.840B.300d.zip
python scripts/prepare_embedding.py \
    -i "emb_dir/glove.840B.300d.txt"
rm "$emb_dir/glove.840B.300d.txt"
