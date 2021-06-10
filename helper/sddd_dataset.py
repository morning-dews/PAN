from sklearn.datasets import fetch_mldata
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import torchvision
import helper.pu_learning_dataset as pu_learning_dataset
import pdb
import random

__author__ = 'garrett_local'


def _prepare_sddd_data():
    # mnist = fetch_mldata('MNIST original', data_home='../../../Data/')

    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # mnist = torchvision.datasets.MNIST('../../../Data/',train=True,download=True)
    sddd_f = open("../../Data/SDDD/sddd", "r")
    sddd = sddd_f.readlines()
    all_x = []
    for sd_x in sddd:
        temp_x = np.array(sd_x.replace('\n', '').split())
        all_x.append(temp_x.astype(np.float).reshape(1, -1))
    
    all_x = np.concatenate(all_x)

    x_range = list(range(all_x.shape[0]))
    random.shuffle(x_range)
    # pdb.set_trace()
    # all_x = all_x[x_range]
    x = np.asarray(all_x[x_range, :-1], dtype=np.float32)
    y = np.asarray(all_x[x_range, -1], dtype=np.float32).reshape(-1)
    
    # pdb.set_trace()

    x = x / x.max(axis=0)


    
    train_x = np.asarray(x[:50000], dtype=np.float32)
    train_y = np.asarray(y[:50000], dtype=np.int32)
    test_x = np.asarray(x[50000:], dtype=np.float32)
    test_y = np.asarray(y[50000:], dtype=np.int32)

    # pdb.set_trace()
    # Binarize labels.
    # train_x = train_x[train_y < 3]
    # train_y = train_y[train_y < 3]
    # test_x = test_x[test_y < 3]
    # test_y = test_y[test_y < 3]
    # pdb.set_trace()

    train_y[train_y % 2 == 1] = -1
    train_y[train_y % 2 == 0] = 1
    train_y[train_y == -1] = 0
    test_y[test_y % 2 == 1] = -1
    test_y[test_y % 2 == 0] = 1
    test_y[test_y == -1] = 0
    # pdb.set_trace()
    return train_x, train_y, test_x, test_y


class SdddDataset(pu_learning_dataset.PuLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_sddd_data()
        super(SdddDataset, self).__init__(*args, **kwargs)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y


class SdddPnDataset(pu_learning_dataset.PnLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_mnist_data()
        super(MnistPnDataset, self).__init__(*args, **kwargs)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y
