import tensorflow as tf


def aumenta_dataset(ds: tf.data.Dataset, shuffle: bool = True, seed: int = 42):
    if shuffle:
        ds = ds.shuffle(buffer_size=len(ds), seed=seed)
    return ds