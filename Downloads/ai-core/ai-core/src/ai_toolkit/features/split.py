import tensorflow as tf
from typing import Tuple


def split_train_val(
    ds: tf.data.Dataset, val_pct: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Divide dataset em treino/validação preservando ordem."""
    ds = ds.enumerate()
    total = tf.data.experimental.cardinality(ds).numpy()
    val_size = int(total * val_pct)
    val = ds.take(val_size).map(lambda idx, data: data)
    train = ds.skip(val_size).map(lambda idx, data: data)
    return train, val