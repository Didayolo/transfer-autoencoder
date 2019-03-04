import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
INGESTION_PATH = '/home/adrien/Documents/autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset

GLOBAL_SHAPE = (1, 52, 52, 3)
GLOBAL_PADDING = False # Padding currently does not work for every dataset (TODO)

def load_dataset(input_dir, basename, batch_size=1, shape=GLOBAL_SHAPE, padding=GLOBAL_PADDING):
    # Corrections of input_dir and basename
    input_dir = os.path.join(input_dir, basename)
    basename = basename + '.data' # why?
    D_train = AutoDLDataset(os.path.join(input_dir, basename, 'train'))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, 'test'))
    x_train, y_train = input_fn(D_train.get_dataset(), batch_size=batch_size,shape=shape, padding=padding)
    x_test, y_test = input_fn(D_test.get_dataset(), batch_size=batch_size, shape=shape, padding=padding)
    return x_train, y_train, x_test, y_test

def processing(tensor_4d, label, shape=GLOBAL_SHAPE, padding=GLOBAL_PADDING):
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

def input_fn(dataset, perform_shuffle=False, batch_size=1, shape=GLOBAL_SHAPE, padding=GLOBAL_PADDING):
    dataset = dataset.map(lambda x, y: processing(x, y, shape=shape, padding=padding))
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat()  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def show_example(tensor_4d, labels=None):
    input_shape = tensor_4d.shape
    print(input_shape)
    shape = (input_shape[1], input_shape[2], 3)
    if labels is not None:
        print(labels)
    plt.imshow(tensor_4d.reshape(shape))
