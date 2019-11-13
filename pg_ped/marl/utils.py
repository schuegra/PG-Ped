# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:47:59 2019

@author: Philipp
"""

from collections import namedtuple

from typing import Tuple, NamedTuple, List

from math import pi as math_pi, floor

import matplotlib.pyplot as plt

import numpy
from scipy.spatial import distance_matrix as scipy_distance_matrix

from PIL import Image

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.nn as nn

from pg_ped.visualization.animator import vis_state, init_axes
from pg_ped.visualization.visualize_cnn import vis_feature_maps, vis_fov
from pg_ped.utils import break_if_nan
from pg_ped.environment_construction.geometry import angle_2D_full

# MISC
Transition = namedtuple('Transition',
                        ('state', 'action', 'action_index', 'log_prob', 'prob', 'next_state',
                         'reward', 'state_value', 'next_state_value'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def distance_matrix(positions: Tensor) -> Tensor:
    """
        Computes the distance for each pair of positions.

        Exploit the symmetry of the distance matrix.

        Parameters:
        -----------

        positions: Tensor
            Has shape (n,2). It represents locations in 2D cartesian coordinates.

    """
    pos = positions.cpu().numpy()
    dists = scipy_distance_matrix(pos, pos)
    return dists


def density(positions: Tensor,
            agent_identity: int,
            influence_radius: float,
            standard_deviation: float) -> float:
    """
        Computes the density based on the distances to other agents.
        The own density is not added.
        The function has a lot of overhead right now, as it computes all densities,
        but only the density at the position of the agent with the given id is
        required.

        Parameters:
        -----------

        positions: Tensor
            Has shape (n,2). It represents locations in 2D cartesian coordinates.
        influence_radius: float
            Positions with greater distance from a point than this radius have
            no effect on the density at a point.

    """

    dists = distance_matrix(positions)
    masked_dists = numpy.where(dists > influence_radius, 0., dists)
    densities = numpy.exp(-masked_dists / standard_deviation ** 2)
    densities = numpy.where(densities < 1. - 1e-10, densities, 0.)
    densities = densities.sum(axis=0)
    return densities[agent_identity]


def reverse_tensor(tensor: Tensor) -> Tensor:
    tensor = torch.rand(10)  # your tensor
    # create inverted indices
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    return tensor.index_select(0, idx)


def split_samples(samples: List[NamedTuple]) -> Tuple[List[Tensor]]:
    try:
        batch = Transition(*zip(*samples))
    except Exception as e:
        print('The number of interactions varies: The runner has one interaction more, as the loop \n'
              'is broken when the runner finishes. If the runner only needs one step (which is not \n'
              'expected behavior most of the times) the batch will be empty.')
        raise (e)
    fields = tuple()
    for field in batch:
        fields += (list(field),)
    return fields


def cart_to_img(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float,
                rows: int, cols: int, person_radius: float = 0.) -> Tuple[float]:
    row = rows * (1 - 1e-5 - torch.abs((y - person_radius) / (y_max - y_min - (2 * person_radius))))
    col = cols * torch.abs(((x - person_radius) / (x_max - x_min - (2 * person_radius)) - 1e-5))
    return torch.floor(abs(row)).int(), torch.floor(abs(col)).int()


def cart_to_img_numpy(xy: numpy.ndarray, x_min: float, x_max: float, y_min: float, y_max: float,
                      rows: int, cols: int) -> Tuple[int]:
    '''
        Convert cartesian coordinates to image coordinates for a Tensor in cartesian coordinates.
    '''
    return (rows * (1 - xy[1] / (y_max - y_min))).astype(numpy.int), (cols * xy[0] / (x_max - x_min)).astype(numpy.int)


def cart_to_img_torch(xy: torch.Tensor, x_min: float, x_max: float, y_min: float, y_max: float,
                      rows: int, cols: int, person_radius: float = 0.) -> torch.Tensor:
    '''
        Convert cartesian coordinates to image coordinates for a Tensor in cartesian coordinates.
    '''
    return (rows * (1 - xy[1] / (y_max - y_min))).int(), (cols * xy[0] / (x_max - x_min)).int()


def rotation_matrix_from_angle(a: Tensor):
    '''
        Generate the matrix which rotates a point in 2D by an angle a if multiplied from the left
        with an 2xN Tensor.
    '''
    R = torch.tensor([[torch.cos(a), torch.sin(a)],
                      [-torch.sin(a), torch.cos(a)]])
    return R


def rotate_image(img: Tensor, angle: float, center_u: Tensor, center_v: Tensor,
                 pad_fill=(int(0.2 * 255), int(0.2 * 255), int(0.2 * 255))):
    '''
        Use torchvision and PIL to rotate a tensorimage.

        angle: rad
    '''

    a_deg = angle * 180. / math_pi - 180
    center = ((img.shape[1] - 1) / 2, (img.shape[2] - 1) / 2)
    pos = (center_u.cpu().numpy(), center_v.cpu().numpy())
    shift = (center[0] - pos[0], center[1] - pos[1])
    shift = (int(shift[0]), int(shift[1]))
    scale = img.max()
    pad = max(shift)

    # Transforms
    toT = T.ToTensor()
    toP = T.ToPILImage()

    # Process
    rot_img = img.cpu()  # PIL only works on cpu
    rot_img = rot_img.float()
    rot_img = toP(rot_img)  # convert to PIL
    rot_img = TF.pad(rot_img, padding=pad, fill=pad_fill)  # pad to avoid loss of pixels
    rot_img = TF.affine(rot_img, translate=[shift[1], shift[0]], angle=0., scale=1.,
                        shear=0.)  # translate position into center
    rot_img = TF.rotate(rot_img, a_deg)  # , center=center)  # rotate around agent
    rot_img = TF.crop(rot_img, pad, pad, img.shape[1], img.shape[2])  # crop away padding
    rot_img = toT(rot_img)  # convert back to tensor
    rot_img = rot_img.to(img.device)

    return rot_img


def cut_window(img: Tensor, rows, cols):
    '''
        Crop a subimage with the same center as img with size rows x cols.
    '''

    center = ((img.shape[1] - 1) / 2, (img.shape[2] - 1) / 2)
    rows_radius, col_radius = (rows - 1) / 2, (cols - 1) / 2
    crop_point = (center[0] + 5 - rows, center[1] - col_radius)

    # Transforms
    toT = T.ToTensor()
    toP = T.ToPILImage()

    # Process
    crop_img = img.cpu()
    crop_img = crop_img.float()
    crop_img = toP(crop_img)  # convert to PIL
    # crop_img = TF.affine(crop_img, angle=0., translate=shift, scale=1., shear=0.)
    crop_img = TF.crop(crop_img, crop_point[0], crop_point[1], rows, cols)  # crop away padding
    crop_img = toT(crop_img)  # convert back to tensor

    return crop_img


def pad_image(img: Tensor, padding: Tuple[int]):
    '''
        Use torchvision and PIL to rotate a tensorimage.

        angle: rad
    '''

    # Transforms
    toT = T.ToTensor()
    toP = T.ToPILImage()

    # Process
    pad_img = img.cpu()  # PIL only works on cpu
    pad_img = pad_img.float()
    pad_img = toP(pad_img)  # convert to PIL
    pad_img = TF.pad(pad_img, padding=padding)
    pad_img = toT(pad_img)  # convert back to tensor
    pad_img = pad_img.to(img.device)

    return pad_img


def angle_img(du, dv, shift_row, shift_col):
    '''
        Compute the angle between to vectors ([du, dv] and [shift_row, shift_col]) in an image.
    '''
    angle = torch.acos(
        (shift_row * du + shift_col * dv) / (
                torch.sqrt(shift_row ** 2 + shift_col ** 2) * torch.sqrt(du ** 2 + dv ** 2) + 1e-8)
    )
    return angle


def normalize_tensor(tensor: Tensor) -> Tensor:
    out_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)
    return out_tensor





def get_gaussian_kernel(standard_deviation: float, influence_radius: float, backward_view: int, device: str):
    # Set these to whatever you want for your gaussian filter
    kernel_size = int(2 * influence_radius + 1)

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = standard_deviation ** 2.

    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1. / (2. * math_pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(backward_view, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=backward_view, out_channels=backward_view,
                                kernel_size=kernel_size, groups=backward_view, bias=False,
                                stride=1, padding=(int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)))

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter.to(device)





def fig_to_data(fig):
    # """Convert matplotlib figure to numpy array with shape rows x cols x 3
    #
    # """
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    # buf = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    # buf.shape = (3, w, h)

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data





def create_circular_mask(h, w, center, radius):
    Y, X = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_ogrid_torch(h, w, device):
    Y, X = numpy.ogrid[:h, :w]
    Y, X = torch.from_numpy(Y).to(device).float(), \
           torch.from_numpy(X).to(device).float()
    return Y, X


def create_circular_mask_torch(h, w, center_u, center_v, radius):
    Y, X = create_ogrid_torch(h, w, center_u.device)
    dist_from_center = torch.sqrt((Y - center_u) ** 2 + (X - center_v) ** 2)
    return torch.where(
        dist_from_center <= radius.float(),
        torch.ones(1, device=center_u.device),
        torch.zeros(1, device=center_u.device)
    )


def create_inverse_circular_mask_torch(h, w, center_u, center_v, radius):
    Y, X = create_ogrid_torch(h, w, center_u.device)
    dist_from_center = torch.sqrt((Y - center_u) ** 2 + (X - center_v) ** 2)
    return torch.where(
        dist_from_center <= radius.float(),
        torch.zeros(1, device=center_u.device),
        torch.ones(1, device=center_u.device)
    )


def create_angular_mask_torch(h, w, center_u, center_v, du, dv, angle_range):
    rows, cols = create_ogrid_torch(h, w, center_u.device)
    shift_row, shift_col = rows - center_u, cols - center_v
    angle = angle_img(du, dv, shift_row, shift_col)

    return torch.where(
        angle <= angle_range,
        torch.ones(1, device=center_u.device),
        torch.zeros(1, device=center_u.device)
    )


def create_inverse_angular_mask_torch(h, w, center_u, center_v, du, dv, angle_range):
    rows, cols = create_ogrid_torch(h, w, center_u.device)
    shift_row, shift_col = rows - center_u, cols - center_v
    angle = torch.acos(
        (shift_row * du + shift_col * dv) / (
                torch.sqrt(shift_row ** 2 + shift_col ** 2) * torch.sqrt(du ** 2 + dv ** 2) + 1e-8)
    )

    return torch.where(
        angle <= angle_range,
        torch.zeros(1, device=center_u.device),
        torch.ones(1, device=center_u.device)
    )


class OrnUhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = numpy.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = numpy.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * numpy.random.randn(self.n_actions)
        self.X += dX
        return self.X