import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
INGESTION_PATH = '/home/adrien/Documents/autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset

def load_dataset(input_dir, basename, shape, padding=False, batch_size=1, train_test_split=False, test_size=5000):
    # Corrections of input_dir and basename
    input_dir = os.path.join(input_dir, basename)
    basename = basename + '.data' # why?
    D_train = AutoDLDataset(os.path.join(input_dir, basename, 'train'))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, 'test'))
    if train_test_split:
        data = D_train.get_dataset()
        data_test = data.take(test_size)
        data_train = data.skip(test_size)
    else:
        data_train = D_train.get_dataset()
        data_test = D_test.get_dataset()
    x_train, y_train = input_fn(data_train, batch_size=batch_size,shape=shape, padding=padding)
    x_test, y_test = input_fn(data_test, batch_size=batch_size,shape=shape, padding=padding)
    return x_train, y_train, x_test, y_test

def processing(tensor_4d, label, shape, padding=False):
    # Process data shape
    # Resize and pad tensor to a certain shape e.g. (1, 50, 50, 3)
    time, row, col, channel = 0, 1, 2, 3
    tensor_shape = tensor_4d.shape

    # if too big, resize
    if (tensor_shape[row] > shape[row] or tensor_shape[col] > shape[col]) or not padding:
        tensor_4d = tf.image.resize_images(tensor_4d, (shape[row], shape[col]))

    # if one channel, gray-scale
    if tensor_4d.shape[channel] == 1:
        tensor_4d = tf.image.grayscale_to_rgb(tensor_4d)

    # padding
    if padding:
        tensor_shape = tensor_4d.shape
        constant_values = 0
        paddings = [[0, m - tensor_shape[i]] for (i,m) in enumerate(shape)]
        tensor_4d = tf.pad(tensor_4d, paddings, 'CONSTANT', constant_values=constant_values)

    assert(tensor_4d.shape == shape)

    return tensor_4d, label

def input_fn(dataset, shape, perform_shuffle=False, batch_size=1, padding=False):
    dataset = dataset.map(lambda x, y: processing(x, y, shape=shape, padding=padding))
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat()  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def tensor_to_array(tensor):
    return tf.Session().run(tensor)

def show_example(tensor_4d, labels=None):
    input_shape = tensor_4d.shape
    print('Shape:', input_shape)
    shape = (input_shape[1], input_shape[2], 3)
    if labels is not None:
        print('Label:', labels)

    # if image example
    show_image(tensor_4d)

def show_image(tensor_4d):
  """Visualize a image represented by `tensor_4d` in RGB or grayscale."""
  num_channels = tensor_4d.shape[-1]
  image = np.squeeze(tensor_4d[0])
  # If the entries are float but in [0,255]
  if not np.issubdtype(image.dtype, np.integer) and np.max(image) > 100:
    image = image / 256
  if num_channels == 1:
    plt.imshow(image, cmap='gray')
  else:
    if not num_channels == 3:
      raise ValueError("Expected num_channels = 3 but got {} instead."\
                       .format(num_channels))
    plt.imshow(image)
  plt.show()

def get_example(input_dir, basename, shape, padding=False, train_test_split=False):
    # Load a dataset
    x_train, y_train, x_test, y_test = load_dataset(input_dir, basename, shape,
                                                  batch_size=256,
                                                  padding=padding, train_test_split=train_test_split)
    with tf.Session() as sess:
        return sess.run(x_test), sess.run(y_test)
