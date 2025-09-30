"""
ai_core.features.tensor
-----------------------
Conversão de features em TensorFlow Dataset.
"""

import tensorflow as tf


def build_tf_dataset(features: dict, labels, batch_size: int = 32):
    """Cria tf.data.Dataset a partir de dict de features + labels."""
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)