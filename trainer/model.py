"""
    EmoRecCom model
"""

import efficientnet.keras as efn
import tensorflow as tf
from transformers import TFAutoModel


def get_img_model(img_model: str):
    models_dict = {
        "efn_b0": efn.EfficientNetB0(include_top=False, weights="noisy-student"),
        "efn_b1": efn.EfficientNetB1(include_top=False, weights="noisy-student"),
        "efn_b2": efn.EfficientNetB2(include_top=False, weights="noisy-student"),
        "efn_b3": efn.EfficientNetB3(include_top=False, weights="noisy-student"),
        "efn_b4": efn.EfficientNetB4(include_top=False, weights="noisy-student"),
        "efn_b5": efn.EfficientNetB5(include_top=False, weights="noisy-student"),
        "efn_b6": efn.EfficientNetB6(include_top=False, weights="noisy-student"),
        "efn_b7": efn.EfficientNetB7(include_top=False, weights="noisy-student"),
    }
    return models_dict[img_model]


def build_model(img_model, bert_model, image_size, max_len, target_size, n_hiddens):

    """
    ICDAR multimodal model for mixed image and dialog data
    NOTE : https://arxiv.org/pdf/1905.12681.pdf

    """

    # Efficient Net pretrained model

    if img_model:
        img_model = get_img_model(img_model)

        img_input = tf.keras.layers.Input(shape=(image_size, image_size, 3), dtype=tf.float32, name="img_input")
        img_out = img_model(img_input)
        img_pooled = tf.keras.layers.GlobalAveragePooling2D()(img_out)

    # Bert pretrained model

    bert_model = TFAutoModel.from_pretrained(bert_model)

    bert_input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_input_id")
    bert_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_attention_mask")
    bert_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="bert_token_type_ids")

    bert_sequence_output = bert_model(
        bert_input_word_ids,
        attention_mask=bert_attention_mask,
        token_type_ids=bert_token_type_ids,
        output_hidden_states=True,
        output_attentions=True,
    )
    if n_hiddens == -1:  # get [CLS] token embedding only
        print("Get pooler output of shape (batch_size, hidden_size)")
        bert_sequence_output = bert_sequence_output[0][:, 0, :]
    else:  # concatenate n_hiddens final layer
        print(f"Concatenate {n_hiddens} hidden_states of shape (batch_size, hidden_size)")
        bert_sequence_output = tf.concat([bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)
        bert_sequence_output = bert_sequence_output[:, 0, :]

    # TODO: implement multimodal model training strategy

    ##################################
    # img_final = tf.keras.layers.Dense(target_size, activation="sigmoid")(img_pooled)
    # bert_final = tf.keras.layers.Dense(target_size, activation="sigmoid")(bert_sequence_output)
    if img_model:
        out = tf.keras.layers.Concatenate()([img_pooled, bert_sequence_output])
        out = tf.keras.layers.Dense(target_size, activation="sigmoid")(out)
    else:
        out = tf.keras.layers.Dense(target_size, activation="sigmoid")(bert_sequence_output)
    ##################################
    # img_output = tf.keras.layers.Dense(8, activation="relu")(img_pooled)
    # bert_output = tf.keras.layers.Dense(8, activation="relu")(bert_sequence_output)
    # out = tf.keras.layers.Concatenate()([img_output, bert_output])
    # out = tf.keras.layers.Dense(target_size, activation="sigmoid")(out)
    ##################################
    if img_model:
        inputs = [img_input, bert_input_word_ids, bert_attention_mask, bert_token_type_ids]
    else:
        inputs = [bert_input_word_ids, bert_attention_mask, bert_token_type_ids]

    model = tf.keras.models.Model(inputs=inputs, outputs=out)

    return model
