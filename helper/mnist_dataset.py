from sklearn.datasets import fetch_mldata
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import torchvision
import helper.pu_learning_dataset as pu_learning_dataset

__author__ = 'garrett_local'


def _prepare_mnist_data():
    mnist = fetch_mldata('MNIST original', data_home='~/Dataset/')
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # mnist = torchvision.datasets.MNIST('../../../Data/',train=True,download=True)
    x = mnist.data
    y = mnist.target
    x = np.reshape(x, (x.shape[0], 28, 28, 1)) / 255.
    train_x = np.asarray(x[:60000], dtype=np.float32)
    train_y = np.asarray(y[:60000], dtype=np.int32)
    test_x = np.asarray(x[60000:], dtype=np.float32)
    test_y = np.asarray(y[60000:], dtype=np.int32)

    # Binarize labels.
    train_y[train_y % 2 == 1] = -1
    train_y[train_y % 2 == 0] = 1
    train_y[train_y == -1] = 0
    test_y[test_y % 2 == 1] = -1
    test_y[test_y % 2 == 0] = 1
    test_y[test_y == -1] = 0
    return train_x, train_y, test_x, test_y


class MnistDataset(pu_learning_dataset.PuLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_mnist_data()
        super(MnistDataset, self).__init__(*args, **kwargs)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y


class MnistPnDataset(pu_learning_dataset.PnLearningDataSet):

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
