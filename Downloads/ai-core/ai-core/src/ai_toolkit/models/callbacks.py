"""Callbacks e augmentações ligadas ao canal NCM.

• ClassWeightAdjuster – callback que ajusta ``class_weight`` dinamicamente a
  cada época, conforme o F1‑score de validação de cada classe.
• zerar_ncm / embaralhar_ncm – rotinas de data‑augmentation específicas para
  o vetor NCM tokenizado. Ambas operam sobre tensores **tf.Tensor** e devem
  ser chamadas **antes** de criar o ``tf.data.Dataset``.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Dynamic class‑weights callback
# ---------------------------------------------------------------------------

class ClassWeightAdjuster(Callback):
    """Ajusta ``class_weight`` se F1 estiver abaixo de certos thresholds.

    A lógica replica o notebook: se F1 < 0.70 multiplica por 1.7, < 0.85 ×1.4,
    < 0.95 ×1.2, respeitando um teto em ``max_class_weights``.

    Args:
        X_val_dict: dicionário de features da validação (id + mask), conforme
                     esperado pelo modelo.
        y_val: labels inteiros de validação.
        class_weights: dict label→peso passado a ``model.fit``.
        max_class_weights: dict label→peso máximo permitido.
        label_encoder: (opcional) scikit ``LabelEncoder`` para logs legíveis.
    """

    def __init__(
        self,
        X_val_dict: Dict[str, tf.Tensor | np.ndarray],
        y_val: np.ndarray,
        class_weights: Dict[int, float],
        max_class_weights: Dict[int, float],
        label_encoder=None,
        thresholds: Tuple[Tuple[float, float], ...] = (
            (0.70, 1.7),
            (0.85, 1.4),
            (0.95, 1.2),
        ),
    ) -> None:
        super().__init__()
        self.X_val = X_val_dict
        self.y_val = y_val
        self.class_weights = class_weights
        self.max_class_weights = max_class_weights
        self.label_encoder = label_encoder
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):  # noqa: D401
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_cls = np.argmax(y_pred, axis=1)
        report = classification_report(
            self.y_val, y_pred_cls, output_dict=True, zero_division=1
        )

        for cls in self.class_weights.keys():
            f1 = report.get(str(cls), {}).get("f1-score", 1.0)
            factor = None
            for thr, mult in self.thresholds:
                if f1 < thr:
                    factor = mult
                    break
            if factor is None:
                continue  # F1 >= 0.95, não mexe

            old = self.class_weights[cls]
            new = min(old * factor, self.max_class_weights[cls])
            self.class_weights[cls] = new

            label_name = (
                self.label_encoder.inverse_transform([cls])[0]
                if self.label_encoder is not None
                else cls
            )
            log.info(
                "ClassWeightAdjuster: %s f1=%.2f → peso %.2f",
                label_name,
                f1,
                new,
            )

# ---------------------------------------------------------------------------
# 2. NCM‑specific augmentations
# ---------------------------------------------------------------------------


def _get_zero_tensor(shape: Tuple[int, int]) -> tf.Tensor:
    return tf.zeros(shape, dtype=tf.int32)


def zerar_ncm(
    X_ncm_ids: tf.Tensor,  # shape (N, max_len_ncm)
    y_labels: tf.Tensor,   # shape (N,)
    zero_pct: float = 0.03,
    exclude_labels: Iterable[int] | None = None,
) -> tf.Tensor:
    """Zera o vetor NCM de *zero_pct* dos exemplos de cada classe (exceto excluídos)."""
    updated = tf.identity(X_ncm_ids)
    unique_labels, counts = tf.unique_with_counts(y_labels)
    zero_vec = _get_zero_tensor((1, X_ncm_ids.shape[1]))
    for label, count in zip(unique_labels.numpy(), counts.numpy()):
        if exclude_labels and label in exclude_labels:
            continue
        n_mod = int(np.ceil(zero_pct * count))
        idx = tf.squeeze(tf.where(y_labels == label))
        sel = tf.random.shuffle(idx)[:n_mod]
        updated = tf.tensor_scatter_nd_update(
            updated,
            tf.expand_dims(sel, 1),
            tf.repeat(zero_vec, repeats=n_mod, axis=0),
        )
    return updated


def embaralhar_ncm(
    X_ncm_ids: tf.Tensor,
    y_labels: tf.Tensor,
    shuffle_pct: float = 0.03,
    exclude_labels: Iterable[int] | None = None,
) -> tf.Tensor:
    """Toma *shuffle_pct* dos vetores NCM de cada classe e embaralha entre elas."""
    indices_all, values_all = [], []
    unique_labels, counts = tf.unique_with_counts(y_labels)
    zero_vec = _get_zero_tensor((1, X_ncm_ids.shape[1]))
    mask_nonzero = tf.reduce_any(tf.not_equal(X_ncm_ids, zero_vec), axis=1)

    for label, count in zip(unique_labels.numpy(), counts.numpy()):
        if exclude_labels and label in exclude_labels:
            continue
        idx = tf.squeeze(tf.where((y_labels == label) & mask_nonzero))
        n_sel = int(np.ceil(shuffle_pct * tf.size(idx).numpy()))
        sel = tf.random.shuffle(idx)[:n_sel]
        indices_all.append(sel)
        values_all.append(tf.gather(X_ncm_ids, sel))

    if not indices_all:
        return X_ncm_ids

    indices_all = tf.concat(indices_all, axis=0)
    values_all = tf.concat(values_all, axis=0)
    values_all = tf.random.shuffle(values_all)

    return tf.tensor_scatter_nd_update(
        X_ncm_ids,
        tf.expand_dims(indices_all, 1),
        values_all,
    )