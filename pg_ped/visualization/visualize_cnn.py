import math

import matplotlib.pyplot as plt

import numpy

import torch
from torch import Tensor

from pg_ped.utils import normalize_tensor, standardize_tensor


def initialize_ax(ax):
    ax.axis('off')
    ax.set_aspect('equal')
    return ax


def vis(x: numpy.ndarray, cmap):
    number_features, rows, columns = x.shape[1:]
    sqrt_number_features = math.sqrt(number_features)
    number_rows = math.ceil(sqrt_number_features)
    number_columns = math.floor(sqrt_number_features)

    # Generate Plot
    all = numpy.zeros([number_rows * rows, number_columns * columns])
    fig, ax = plt.subplots()

    for row in range(number_rows):
        for col in range(number_columns):
            all[row * rows:(row + 1) * rows, col * columns:(col + 1) * columns] = \
                x[0, col * number_columns + row, :, :]

    ax = initialize_ax(ax)
    ax.imshow(all, cmap=cmap, vmin=0, vmax=1)
    plt.show()


def vis_feature_maps(feature_maps: Tensor, cmap='gist_gray'):
    '''
    Visualize feature maps.
    '''

    feature_maps_array = feature_maps.cpu().detach().numpy()
    vis(feature_maps_array, cmap)


def vis_fov(fov: Tensor):
    '''
    Visualize feature maps.
    '''

    fov_array = fov.cpu().detach().numpy()
    imgs = [
        fov_array[:3],
        fov_array[3:6],
        fov_array[6:9]
    ]

    fig, axs = plt.subplots(1, 3)
    for ax, img in zip(axs, imgs):
        ax.imshow(img.transpose(1, 2, 0))
        ax.axis('off')
    plt.show()


def vis_weights(layer: torch.nn.Module):
    '''
    Visualize weights of a layer
    '''

    filter_array = layer.weight.data.cpu().detach().numpy()
    filter_array = filter_array.reshape(1,
                                        filter_array.shape[0] * filter_array.shape[1],
                                        filter_array.shape[2], filter_array.shape[3])
    vis(filter_array)


def vis_fc(features: Tensor):
    '''
    Visualize the fully connected layers outputs.

    :param scores: Tensor of shape 1 x N (batch size, number of features)
    :return:
    '''

    features_array = features.cpu().detach().numpy()
    features_array = features_array.squeeze(0)

    plt.plot(features_array, '.')
    plt.show()
