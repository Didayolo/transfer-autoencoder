import matplotlib.pyplot as plt
import os
import sys
INGESTION_PATH = '/home/adrien/Documents/autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program/'
sys.path.append(INGESTION_PATH)
from dataset import AutoDLDataset


def load_dataset(input_dir, basename, batch_size=1):
    # Corrections of input_dir and basename
    input_dir = os.path.join(input_dir, basename)
    basename = basename + '.data' # why?
    D_train = AutoDLDataset(os.path.join(input_dir, basename, 'train'))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, 'test'))
    x_train, y_train = input_fn(D_train.get_dataset(), batch_size=batch_size)
    x_test, y_test = input_fn(D_test.get_dataset(), batch_size=batch_size)
    return x_train, y_train, x_test, y_test

def input_fn(dataset, perform_shuffle=False, batch_size=1):
    # maybe?
    # dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat()  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def show_example(tensor_4d, labels=None, n=0, shape=(28,28)):
    input_shape = tensor_4d[n].shape
    print(input_shape)
    if labels is not None:
        print(labels[n])
    plt.imshow(tensor_4d[n].reshape(shape))
