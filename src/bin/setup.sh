#!/bin

pip install -r requirements.txt

[[ -d ../data ]] || ((gdown --id 1OUd7dQybiioKMu7NXtWxIITdun8SaaUX) && (unzip data.zip -d ../))

# [[ -f glove.840B.300d.pkl ]] || (gdown --id 1-7SQ49tYL2STnSkVrgwXwd_wyKkt5oMF)
# [[ -f wiki.en.vec.pkl ]] || (gdown --id 1-XwvKbEpZuIiLRK9O0Oa8OO8BnaC7LS9)
# [[ -f crawl-300d-2M.vec.pkl ]] || (gdown --id 1-Gk8uChzM5Iym9GC_LifzrQCMhodIJXW)
# [[ -f wiki-news-300d-1M.vec.pkl ]] || (gdown --id 1-GhBuFTuTOsX1oreXRhSFF4L9Za4Xl7K)