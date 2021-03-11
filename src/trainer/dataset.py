"""
    Main data generator class
"""


import numpy as np
import tensorflow as tf


class ICDARGenerator(tf.keras.utils.Sequence):

    """
    Data Generator for Keras model
    """

    def __init__(
        self,
        df,
        bert_tokenizer,
        shuffle,
        batch_size,
        max_len,
        tf_tokenizer,
        max_word,
        image_size,
        target_cols,
    ):
        self.shuffle = shuffle
        self.image_size = image_size
        self.target_cols = target_cols
        self.batch_size = batch_size

        self.image_path = df["file_path"].values
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

        self.image_features = image_size > 0
        self.word_embedding_features = tf_tokenizer is not None

        if self.word_embedding_features:
            self.sequences = tf_tokenizer.texts_to_sequences(texts)
            self.sequences = tf.keras.preprocessing.sequence.pad_sequences(
                self.sequences, maxlen=max_word, padding="post", truncating="post"
            )

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

        labels = np.array([self.labels[k] for k in indexes])

        input_ids = tf.convert_to_tensor([self.input_ids[k] for k in indexes])
        attention_mask = tf.convert_to_tensor([self.attention_mask[k] for k in indexes])
        token_type_ids = tf.convert_to_tensor([self.token_type_ids[k] for k in indexes])

        features = [input_ids, attention_mask, token_type_ids]

        if self.image_features:
            image_path = [self.image_path[k] for k in indexes]
            images = []
            for fp in image_path:
                images.append(self.process_image(fp))
            images = tf.stack(images)
            features.append(images)

        if self.word_embedding_features:
            sequences = tf.convert_to_tensor([self.sequences[k] for k in indexes])
            features.append(sequences)

        return features, labels

    def decode(self, path: str):

        file_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(file_bytes, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (self.image_size, self.image_size))

        return image

    def augment(self, image):
        # TODO: add images augmentations
        return image

    def process_image(self, path):
        return self.augment(self.decode(path))
