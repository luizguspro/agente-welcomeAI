import tensorflow as tf
from transformers import TFAutoModel


# -------- Config --------
BERT_ID = "neuralmind/bert-base-portuguese-cased"
TEMPERATURE_LOGITS = 3.0
TEMPERATURE_FONETICA = 2.0


# --------  Helpers --------
def bert_branch(name: str, max_len: int, trainable_layers: int = 1):
    """
    Cria inputs e o branch BERT com pooling custom.

    Args:
        name: identificador para camadas e inputs.
        max_len: comprimento máximo da sequência de tokens.
        trainable_layers: número de últimos layers do encoder que ficam treináveis.

    Returns:
        inputs: [ids_input, mask_input]
        pooled: saída pós-pooling (batch, hidden_size)
    """
    ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name=f"{name}_ids")
    masks = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name=f"{name}_mask")

    bert = TFAutoModel.from_pretrained(BERT_ID, name=f"bert_{name}")
    # Congela todas as camadas exceto as últimas trainable_layers
    for layer in bert.bert.encoder.layer[:-trainable_layers]:
        layer.trainable = False

    seq = bert([ids, masks]).last_hidden_state  # (batch, seq_len, hidden)
    # pooling via atenção simples: soma ponderada
    attn = tf.keras.layers.Dense(1, activation="tanh")(seq)
    weights = tf.keras.layers.Softmax(axis=1)(attn)
    pooled = tf.keras.layers.Multiply()([seq, weights])
    pooled = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name=f"attn_{name}")(pooled)

    return [ids, masks], pooled


def build_model_logits(max_len_desc: int, max_len_ncm: int, n_labels: int):
    """
    Constrói o modelo baseline (logits) combinando branches de descrição e NCM.
    """
    # branch NCM
    in_ncm, b_ncm = bert_branch("ncm", max_len_ncm)
    # branch Descrição
    in_desc, b_desc = bert_branch("descricao", max_len_desc)

    x = tf.keras.layers.Concatenate(name="concat_ncm_desc")([b_ncm, b_desc])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(
        n_labels,
        activation=lambda z: tf.nn.softmax(z / TEMPERATURE_LOGITS),
        name="softmax_logits"
    )(x)

    return tf.keras.Model(inputs=in_ncm + in_desc, outputs=out, name="model_logits")


def build_model_fonetica(max_len_fon: int, max_len_desc: int, n_labels: int):
    """
    Constrói o modelo fonética+descrição (alfa) para geração de soft-labels.
    """
    # branch Fonética
    in_fon, b_fon = bert_branch("fonetica", max_len_fon)
    # branch Descrição
    in_desc, b_desc = bert_branch("descricao", max_len_desc)

    x = tf.keras.layers.Concatenate(name="concat_fon_desc")([b_fon, b_desc])
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(
        n_labels,
        activation=lambda z: tf.nn.softmax(z / TEMPERATURE_FONETICA),
        name="softmax_fonetica"
    )(x)

    return tf.keras.Model(inputs=in_fon + in_desc, outputs=out, name="model_fonetica")


def build_model_augmented(max_len_desc: int, max_len_ncm: int, n_labels: int):
    """
    Constrói o modelo augmented (student) com mesma topologia que logits.
    """
    model = build_model_logits(max_len_desc, max_len_ncm, n_labels)
    model._name = "model_augmented"
    return model


# -------- Trainer --------
def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    loss_fn,
    class_weights: dict = None,
    run_id: str = "run",
    epochs: int = 5,
    lr: float = 1e-5,
    output_dir: str = "artefatos"
):
    """
    Treina e salva checkpoints do modelo.

    Args:
        model: modelo Keras compilado.
        train_ds: tf.data.Dataset de treino.
        val_ds: tf.data.Dataset de validação.
        loss_fn: função de perda ou string (ex: 'sparse_categorical_crossentropy').
        class_weights: mapeamento label->peso para class_weight.
        run_id: identificador da execução para versão de artefatos.
        epochs: número de épocas.
        lr: learning rate inicial.
        output_dir: diretório base para salvar artefatos.
    """
    # callbacks
    ckpt_path = f"{output_dir}/{run_id}/{model.name}.keras"
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # compile
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # fit
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[ckpt, es]
    )

    return model