from typing import List

from math import pi as math_pi

import torch
import torch.nn.functional as F

from pg_ped.utils import standardize_tensor, normalize_tensor
from pg_ped.visualization.visualize_cnn import vis_feature_maps, vis_fc


def conv_size(W, F, P, S):
    return int((W - F + 2 * P) * 1. / S) + 1


class TinyCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float):
        super(TinyCNN, self).__init__()
        self._fc1_input_size = 32
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 64)
        self.fc2 = torch.nn.Linear(64, output_size)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StandardCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(StandardCNN, self).__init__()
        self._fc1_input_size = 64
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 256)
        self.fc2 = torch.nn.Linear(256, output_size)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)
        # x = normalize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DropoutCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float):
        super(DropoutCNN, self).__init__()
        self._fc1_input_size = 128
        conv2_size = conv_size(conv_size(rows, 7, 0, 2), 3, 0, 1)
        conv3_size = conv_size(conv2_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, output_size)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)
        # x = normalize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2d(x)

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        x = self.fc3(x)
        return x


class SimpleCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(SimpleCNN, self).__init__()
        self._fc1_input_size = 256
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)
        x = normalize_tensor(x)

        # Feature Extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeeperCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super().__init__()
        self._fc1_input_size = 256
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        pool3_size = 0.5 * conv_size(conv_size(pool2_size, 3, 0, 1), 3, 0, 1)
        conv7_size = conv_size(pool3_size, 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        # self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        # self.conv7 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.fc2 = torch.nn.Linear(512, output_size)
        self.lrn = torch.nn.LocalResponseNorm(2)

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)
        x = normalize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = self.pool(x)
        # x = F.relu(self.conv7(x))
        # x = self.lrn(x)

        # Classification
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DropoutMLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, dropout_probability: float):
        super().__init__()

        self._output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, output_size)
        self.activation = torch.nn.Tanh()
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        # x = standardize_tensor(x)

        # Inference
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2


class DropoutMLP2(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, dropout_probability: float):
        super().__init__()

        self._output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, output_size)
        self.activation = torch.nn.Tanh()
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Inference
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2


class DropoutMLP3(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, dropout_probability: float):
        super().__init__()

        self._output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)
        self.activation = torch.nn.ReLU()
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.activation(self.fc2(x))
        x = self.dropout1d(x)
        x = self.fc3(x)
        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2


class MLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.fcs = torch.nn.ModuleList()
        self.fcs += [torch.nn.Linear(self._input_size, n_neurons[0])]
        for l in range(n_hidden - 1):
            self.fcs += [torch.nn.Linear(n_neurons[l], n_neurons[l + 1])]
        self.fcs += [torch.nn.Linear(n_neurons[-1], self._output_size)]

        self.activation = activation
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def reconstruct(self, n_hidden: int, n_neurons: List[int],
                    activation: torch.nn.Module, dropout_probability: float):

        self.fcs = torch.nn.ModuleList()
        self.fcs += [torch.nn.Linear(self._input_size, n_neurons[0])]
        for l in range(n_hidden - 1):
            self.fcs += [torch.nn.Linear(n_neurons[l], n_neurons[l + 1])]
        self.fcs += [torch.nn.Linear(n_neurons[-1], self._output_size)]

        self.activation = activation
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):

        for linear in self.fcs[:-1]:
            x = self.activation(linear(x))
            x = self.dropout1d(x)
        x = self.fcs[-1](x)
        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2


class PedDomainMLP(torch.nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_agents: int,
                 dropout_probability: float):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._n_agents = n_agents

        self.fc_agent = torch.nn.Linear(3, 32)
        self.fc_others = torch.nn.Linear(3, 32)
        self.fc_obstacles = torch.nn.Linear(2, 32)
        self.fuse1 = torch.nn.Linear((n_agents + 4) * 32, 128)
        #self.fuse2 = torch.nn.Linear(256, 64)
        self.out = torch.nn.Linear(128, output_size)

        self.activation = torch.nn.ReLU()
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

        self.agent_fields = [0, 1, 2]
        self.other_agent_fields = [i for i in range(3, 3 * n_agents)]
        self.obstacle_fields = [i for i in range(3 * n_agents, 3 * n_agents + 2 * 4)]

    def forward(self, x):
        #x = self.standardize_kinematics(x)

        x_agent = x[:, self.agent_fields]
        x_others = x[:, self.other_agent_fields]
        x_obstacles = x[:, self.obstacle_fields]

        fc_agent = self.activation(self.fc_agent(x_agent))

        fc_others = []
        for i in range(self._n_agents - 1):
            fc_others += [self.activation(self.fc_others(x_others[:, 3 * i: 3 * (i + 1)]))]
        fc_others = torch.cat(fc_others, dim=1)

        fc_obstacles = []
        for i in range(4):
            fc_obstacles += [self.activation(self.fc_obstacles(x_obstacles[:, 2 * i: 2 * (i + 1)]))]
        fc_obstacles = torch.cat(fc_obstacles, dim=1)

        fc_all = torch.cat([fc_agent, fc_others, fc_obstacles], dim=1)
        x = self.activation(self.fuse1(fc_all))
        #x = self.activation(self.fuse2(x))
        x = self.out(x)

        return x


class NPedMLP(torch.nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_agents: int,
                 dropout_probability: float):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._n_agents = n_agents

        self.fc_agent = torch.nn.Linear(3, 8)
        self.fc_agent2 = torch.nn.Linear(8, 8)
        self.fc_others = torch.nn.Linear(3, 8)
        self.fc_obstacles = torch.nn.Linear(2, 8)

        self.fuse_others = torch.nn.Linear((n_agents - 1) * 8, 32)
        self.fuse_obstacles = torch.nn.Linear(4 * 8, 16)

        self.fuse1 = torch.nn.Linear(56, 128)
        #self.fuse2 = torch.nn.Linear(128, 256)
        self.out = torch.nn.Linear(128, output_size)
        #self.out = torch.nn.Linear(256, output_size)

        self.activation = torch.nn.ReLU()
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

        self.agent_fields = [0, 1, 2]
        self.other_agent_fields = [i for i in range(3, 3 * n_agents)]
        self.obstacle_fields = [i for i in range(3 * n_agents, 3 * n_agents + 2 * 4)]

    def forward(self, x):
        x = self.normalize_kinematics(x)

        x_agent = x[:, self.agent_fields]
        x_others = x[:, self.other_agent_fields]
        x_obstacles = x[:, self.obstacle_fields]

        fc_agent = self.activation(self.fc_agent(x_agent))
        fc_agent = self.activation(self.fc_agent2(fc_agent))

        fc_others = []
        for i in range(self._n_agents - 1):
            fc_others += [self.activation(self.fc_others(x_others[:, 3 * i: 3 * (i + 1)]))]
        fc_others = self.activation(self.fuse_others(torch.cat(fc_others, dim=1)))

        fc_obstacles = []
        for i in range(4):
            fc_obstacles += [self.activation(self.fc_obstacles(x_obstacles[:, 2 * i: 2 * (i + 1)]))]
        fc_obstacles = self.activation(self.fuse_obstacles(torch.cat(fc_obstacles, dim=1)))

        fc_all = torch.cat([fc_agent, fc_others, fc_obstacles], dim=1)
        x = self.activation(self.fuse1(fc_all))
        #x = self.activation(self.fuse2(x))
        x = self.out(x)

        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2

    def normalize_kinematics(self, x,
                             min_dist=0., max_dist=23.6524 ** 0.5,
                             min_angle=0. , max_angle=2 * math_pi,
                             min_velocity=0., max_velocity=1.8):
        n_agents = self._n_agents
        dists = [2] + [i for i in range(4, 3 * (n_agents - 1) + 2, 3)] + [i for i in range(-8, 0, 2)]
        angles = [1] + [i for i in range(5, 3 * (n_agents - 1) + 3, 3)] + [i for i in range(-7, 0, 2)]
        velocities = [0] + [i for i in range(3, 3 * (n_agents - 1) + 1, 3)]

        x[:, dists] = (x[:, dists] - min_dist) / (max_dist - min_dist)
        x[:, angles] = (x[:, angles] - min_angle) / (max_angle - min_angle)
        x[:, velocities] = (x[:, velocities] - min_velocity) / (max_velocity - min_velocity)

        return x
