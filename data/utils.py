import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset(name="year_1_score_density", batch_size=1, split="100%", shuffle_files=False):
  ds = tfds.load(name, split='train[:{}]'.format(split), shuffle_files=shuffle_files)
  ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return iter(tfds.as_numpy(ds))

