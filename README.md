<div align="center">

## [ICDAR 2021](https://icdar2021.org/program-2/competitions/) <img src="assets/icdar.png" alt="ICDAR 2021" width="25" height="15">

### [Multimodal Emotion Recognition on Comics scenes](https://competitions.codalab.org/competitions/27884)

</div>

### 🏹 Usage:
```
usage: main.py [-h] [--train_dir TRAIN_DIR] [--test_dir TEST_DIR]
               [--output_dir OUTPUT_DIR]
               [--target_cols TARGET_COLS [TARGET_COLS ...]]
               [--gpus GPUS [GPUS ...]] [--do_train] [--do_infer]
               [--ckpt_dir CKPT_DIR] [--image_model IMAGE_MODEL]
               [--bert_model BERT_MODEL] [--word_embedding WORD_EMBEDDING]
               [--max_vocab MAX_VOCAB] [--max_word MAX_WORD]
               [--image_size IMAGE_SIZE] [--max_len MAX_LEN] [--lower]
               [--text_separator TEXT_SEPARATOR] [--n_hiddens N_HIDDENS]
               [--drop_rate DROP_RATE] [--lr LR] [--batch_size BATCH_SIZE]
               [--n_epochs N_EPOCHS] [--kaggle] [--seed SEED]

ICDAR 2021: Multimodal Emotion Recognition on Comics scenes (EmoRecCom)

optional arguments:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR path to the train data directory to train model
  --test_dir TEST_DIR   path to the test data directory to predict
  --output_dir OUTPUT_DIR path to directory for models saving
  --target_cols TARGET_COLS [TARGET_COLS ...]
                        define columns for forecasting
  --gpus GPUS [GPUS ...] select gpus to use
  --do_train            whether train the pretrained model with provided train
                        data
  --do_infer            whether predict the provided test data with the
                        trained models from checkpoint directory
  --ckpt_dir CKPT_DIR   path to the directory containing checkpoints (.h5)
                        models
  --image_model IMAGE_MODEL pretrained image model name in list ['efn-b0',
                        'efn-b1', 'efn-b2', 'efn-b3', 'efn-b4', 'efn-b5',
                        'efn-b6', 'efn-b7'] None for using unimodal model
  --bert_model BERT_MODEL
                        path to pretrained bert model path or directory (e.g:
                        https://huggingface.co/models)
  --word_embedding WORD_EMBEDDING path to a pretrained static word embedding in list
                        ['glove.840B.300d', 'wiki.en.vec',
                        'crawl-300d-2M.vec', 'wiki-news-300d-1M.vec'] None for
                        using bert model only to represent text
  --max_vocab MAX_VOCAB maximum of word in the vocabulary (Tensorflow word
                        tokenizer)
  --max_word MAX_WORD   maximum word per text sample (Tensorflow word
                        tokenizer)
  --image_size IMAGE_SIZE size of image
  --max_len MAX_LEN     max sequence length for padding and truncation (Bert
                        word tokenizer)
  --lower               whether lowercase text or not
  --text_separator TEXT_SEPARATOR
                        define separator to join conversations
  --n_hiddens N_HIDDENS concatenate n_hiddens final layer to get sequence's
                        bert embedding, -1 for using [CLS] token embedding
                        only
  --drop_rate DROP_RATE drop out rate for both images and text encoders
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        num examples per batch
  --n_epochs N_EPOCHS   num epochs required for training
  --kaggle              whether using kaggle environment or not
  --seed SEED           seed for reproceduce
```

### 💪 [Outputs](https://drive.google.com/drive/folders/1mfeWRV9-yfmcbIWgLWLBKblM1-cPaWOi?usp=sharing)