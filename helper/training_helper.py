import helper.cifar10_dataset as cifar10_dataset
import helper.mnist_dataset as mnist_dataset
import helper.sddd_dataset as sddd_dataset
import helper.news_dataset as news_dataset
# import network

__author__ = 'garrett_local'


def load_dataset(cfg):
    dataset_name = cfg['dataset']['dataset_name']

    if dataset_name == 'mnist':
        return mnist_dataset.MnistDataset, mnist_dataset.MnistPnDataset
    if dataset_name == 'cifar10':
        return cifar10_dataset.Cifar10Dataset, cifar10_dataset.Cifar10PnDataset
    if dataset_name == 'sddd':
        return sddd_dataset.SdddDataset, sddd_dataset.SdddPnDataset
    if dataset_name == 'sensor':
        return cifar10_dataset.Cifar10Dataset, cifar10_dataset.Cifar10PnDataset
    if dataset_name == 'news':
        return news_dataset.MnistDataset, news_dataset.MnistPnDataset


# def load_network(cfg):
#     network_name = cfg['network']['network_name']
#     if network_name == 'MLP':
#         return network.MultilayerPerceptron
#     if network_name == 'CNN':
#         return network.ConvolutionalNeuralNetwork
