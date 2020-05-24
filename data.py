import tensorflow as tf

from tf.keras.datasets import mnist

def get_mnist_dataset(batch_size=4):
  train, test = mnist.load_data()
  images, labels = train
  images = images/255
  tf.image.resize(images, (32, 32, 1))
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.shuffle(60000, reshuffle_each_iteration=True)
  dataset = dataset.map(train_random_preprocess, num_parallel_calls=4)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.prefetch(1)
  return dataset


def train_random_preprocess(images):
  # rotate thingy
  tf.keras.preprocessing.image.random_rotation(
    x, rg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0,
    interpolation_order=1)
  # flip thingy
  tf.image.random_flip_left_right(image)
  tf.image.random_flip_up_down(image)