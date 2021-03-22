"""
    EmoRecCom model
"""

import efficientnet.keras as efn
import tensorflow as tf
from transformers import TFAutoModel


def get_image_model(image_model: str):
    """
    Get Keras image model by name
    Args:
        image_model (str): Pretrained image model name
    Returns:
        tf.keras.Model: Pretrained image model
    """
    models_dict = {
        "efn-b0": efn.EfficientNetB0(include_top=False, weights="noisy-student"),
        "efn-b1": efn.EfficientNetB1(include_top=False, weights="noisy-student"),
        "efn-b2": efn.EfficientNetB2(include_top=False, weights="noisy-student"),
        "efn-b3": efn.EfficientNetB3(include_top=False, weights="noisy-student"),
        "efn-b4": efn.EfficientNetB4(include_top=False, weights="noisy-student"),
        "efn-b5": efn.EfficientNetB5(include_top=False, weights="noisy-student"),
        "efn-b6": efn.EfficientNetB6(include_top=False, weights="noisy-student"),
        "efn-b7": efn.EfficientNetB7(include_top=False, weights="noisy-student"),
    }
    return models_dict[image_model]


def build_model(
    bert_model,
    max_len,
    n_hiddens,
    image_model,
    image_size,
    target_size,
    max_word,
    embedding_matrix=None,
):
    """
    ICDAR multimodal model for mixed image and dialog data
    NOTE : https://arxiv.org/pdf/1905.12681.pdf
    """

    if image_model is not None:
        assert image_size > 1, f"{image_model} requires image size > 0, got {image_size} instead"
    if embedding_matrix is not None:
        assert max_word > 1, f"Word embedding requires max sequence length > 0, got {max_word} instead"

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

    """
    bert_sequence_output = (last_hidden_state (batch_size, sequence_length, hidden_size),
                            pooler_output (batch_size, hidden_size),
                            hidden_states (batch_size, sequence_length, hidden_size),
                            attentions (batch_size, num_heads, sequence_length, sequence_length)
                            )
    """

    if n_hiddens == -1:  # get [CLS] token embedding only
        bert_sequence_output = bert_sequence_output[0][:, 0, :]
        # bert_sequence_output = bert_sequence_output[1]
    else:  # concatenate n_hiddens final layer
        bert_sequence_output = tf.concat([bert_sequence_output[2][-i] for i in range(n_hiddens)], axis=-1)
        bert_sequence_output = tf.keras.layers.Flatten()(bert_sequence_output)

    inputs = [bert_input_word_ids, bert_attention_mask, bert_token_type_ids]
    outputs = [bert_sequence_output]

    # Efficient Net pretrained model
    if image_model:
        image_model = get_image_model(image_model)
        image_input = tf.keras.layers.Input(shape=(image_size, image_size, 3), dtype=tf.float32, name="image_input")
        image_output = image_model(image_input)
        image_output = tf.keras.layers.GlobalAveragePooling2D()(image_output)
        inputs.append(image_input)
        outputs.append(image_output)

    # Static word embedding pretrained weights
    if embedding_matrix is not None:
        sequence_input = tf.keras.layers.Input(shape=(max_word,), name="sequence_input")
        embedding_sequence = tf.keras.layers.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False
        )(sequence_input)
        embedding_sequence = tf.keras.layers.SpatialDropout1D(0.2)(embedding_sequence)
        embedding_sequence = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1)
        )(embedding_sequence)
        embedding_sequence = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform"
        )(embedding_sequence)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(embedding_sequence)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(embedding_sequence)
        embedding_sequence_output = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        inputs.append(sequence_input)
        outputs.append(embedding_sequence_output)

    if len(outputs) > 1:
        outputs = tf.keras.layers.Concatenate()(outputs)
    else:
        outputs = outputs[0]

    outputs = tf.keras.layers.Dense(target_size, activation="sigmoid")(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
