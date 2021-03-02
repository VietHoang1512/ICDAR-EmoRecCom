"""
    Main data generator class
"""


import numpy as np
import tensorflow as tf


class ICDARGenerator(tf.keras.utils.Sequence):

    """Data Generator for Keras model"""

    def __init__(
        self,
        df,
        multimodal,
        bert_tokenizer,
        shuffle,
        batch_size,
        image_size,
        target_cols,
        max_len,
    ):
        """
        ICDAR dataset constructor

        Args:
            df (DataFrame): Processed dataframe, including image paths and dialogues
            multimodal (bool): whether generating image features or not # FIXME
            bert_tokenizer (AutoTokenizer): transformer's tokenizer class
            shuffle (bool): whether shuffling data or not (set=False) when inferencing test set
            batch_size (int): size of mini-batch
            image_size (int): size of images (image_size, image_size, 3)
            target_cols (list): target columns (some of the following: angry, disgust, fear, happy, sad, surprise, neutral, other)
            max_len (int): max sequence length each batch
        """
        self.shuffle = shuffle
        self.image_size = image_size
        self.target_cols = target_cols
        self.batch_size = batch_size

        self.img_path = df["file_path"].values
        self.labels = df[target_cols].values.astype(float)

        texts = df["text"].tolist()
        bert_encoded = bert_tokenizer.batch_encode_plus(
            texts,
            return_token_type_ids=True,
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        self.input_ids = bert_encoded["input_ids"]
        self.attention_mask = bert_encoded["attention_mask"]
        self.token_type_ids = bert_encoded["token_type_ids"]

        self.total = len(df)
        self.indexes = np.arange(self.total)
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(self.total / self.batch_size))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        img_path = [self.img_path[k] for k in indexes]

        images = []
        for fp in img_path:
            images.append(self.process_image(fp))
        images = tf.stack(images)

        input_ids = tf.convert_to_tensor([self.input_ids[k] for k in indexes])
        attention_mask = tf.convert_to_tensor([self.attention_mask[k] for k in indexes])
        token_type_ids = tf.convert_to_tensor([self.token_type_ids[k] for k in indexes])

        labels = np.array([self.labels[k] for k in indexes])

        return [images, input_ids, attention_mask, token_type_ids], labels

    def decode(self, path):
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, (self.image_size, self.image_size))

        return img

    def augment(self, img):
        # TODO: add images augmentations
        return img

    def process_image(self, path):
        return self.augment(self.decode(path))
