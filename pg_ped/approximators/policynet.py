from typing import List, Tuple
from math import pi
import math

from sklearn.preprocessing import PolynomialFeatures

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18

from pg_ped.utils import standardize_tensor, normalize_tensor, standardize_tensor_nn
from pg_ped.visualization.visualize_cnn import vis_feature_maps, vis_fc, vis_fov





class PolicyNetRecurrent(nn.Module):

    def __init__(self, input_size: int,
                 output_size: int,
                 state_value_function: nn.Module = None,
                 action_value_function: nn.Module = None,
                 number_agents: int = 1,
                 variables_per_agent_per_timestep: int = 4,
                 backward_view: int = 3) -> None:
        super(PolicyNetRecurrent, self).__init__()
        print('Recurrent Policy Net is initialized')

        hidden_size1 = 128
        hidden_size2 = 128
        hidden_size3 = 64
        self._number_agents = number_agents
        self._variables_per_agent_per_timestep = variables_per_agent_per_timestep
        self._backward_view = backward_view

        # Layers
        self.fc1 = nn.Linear(self._number_agents * self._variables_per_agent_per_timestep, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self._prepare(x)
        state_sequence = self._x_to_sequence(x)
        x = state_sequence[0].unsqueeze(0)
        x = F.relu(self.fc1(x))
        for state in state_sequence[1:]:
            x_t = state.unsqueeze(0)
            x_t = self.fc1(x_t)
            x = F.relu(self.fc2(x + x_t))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        out = self.softmax(x)

        return out

    def _prepare(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x).reshape(len(x), x[0].shape[0], x[0].shape[1])

    def _x_to_sequence(self, x: Tensor) -> List[Tensor]:

        sequence = []
        for t in range(self._backward_view):
            x_t = torch.zeros(self._number_agents * self._variables_per_agent_per_timestep, device=x.device)
            for a in range(self._number_agents):
                i = (a * self._variables_per_agent_per_timestep, (a + 1) * self._variables_per_agent_per_timestep)
                j = (t * self._variables_per_agent_per_timestep, (t + 1) * self._variables_per_agent_per_timestep)
                x_t[i[0]:i[1]] = x[0, a, j[0]:j[1]]
            sequence += [x_t]

        return sequence


class PolicyNetForward(nn.Module):

    def __init__(self, input_size: int,
                 output_size: int,
                 number_agents: int = 1,
                 state_value_function: nn.Module = None,
                 action_value_function: nn.Module = None) -> None:
        super(PolicyNetForward, self).__init__()
        print('Policy Net is initialized')

        hidden_size1 = 64
        hidden_size2 = 128
        hidden_size3 = 64

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        # self.fc1 = nn.Linear(input_size, hidden_size3)
        # self.fc4 = nn.Linear(hidden_size3, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self._prepare(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        out = self.softmax(x)
        return out

    def _prepare(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x).reshape(len(x), x[0].shape[0], x[0].shape[1])


class PolicyNetForwardTiny(nn.Module):

    def __init__(self, input_size: int,
                 output_size: int) -> None:
        super(PolicyNetForwardTiny, self).__init__()
        print('Policy Net is initialized')

        hidden_size1 = 32

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self._prepare(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.softmax(x)
        return out

    def _prepare(self, x: List[Tensor]) -> Tensor:
        return torch.cat(x).reshape(len(x), x[0].shape[0], x[0].shape[1])


class TinyCNN(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(TinyCNN, self).__init__()

        conv1_size = 16
        conv2_size = 32
        conv3_size = 64
        fc1_size = 32
        # number_of_pooling_layers = 1
        # self._fc1_input_size = int(conv3_size * int(rows / (2*number_of_pooling_layers)) * int(cols / (2*number_of_pooling_layers)))
        self._fc1_input_size = conv3_size * rows * cols

        # self.conv1 = torch.nn.Conv2d(input_channels, conv1_size, kernel_size=3, stride=1, padding=1)
        # self.conv2 = torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(input_channels, conv3_size, kernel_size=3, stride=1, padding=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)
        print(self)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def conv_size(W, F, P, S):
    return int((W - F + 2 * P) * 1. / S) + 1


class SimpleCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(SimpleCNN, self).__init__()
        self._fc1_input_size = 128
        pool_size = 0.5 * conv_size(conv_size(rows, 5, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = standardize_tensor(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SimpleCNN2(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(SimpleCNN, self).__init__()
        self._fc1_input_size = 512
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(0.5 * conv_size(conv_size(pool_size, 3, 0, 1), 3, 0, 1), 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = standardize_tensor(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))

        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SimpleCNNContinous(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, scale: float = 1.4):
        super(SimpleCNNContinous, self).__init__()
        self._scale = torch.tensor(scale)
        self._fc1_input_size = 128
        self._fc1_input_size *= conv_size(0.5 * conv_size(conv_size(rows, 5, 0, 1), 3, 0, 1), 3, 0, 1) ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = standardize_tensor(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        mean_v = torch.sigmoid(x[:, 0]) * self._scale
        mean_a = torch.tanh(x[:, 1]) * pi / 2.
        std_v = F.elu(x[:, 2]) + 1
        std_a = F.elu(x[:, 3]) + 1
        x = torch.cat([mean_v, mean_a, std_v, std_a])
        return x


class VGG11(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(VGG11, self).__init__()
        self.channel_adaption = torch.nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        self.scores = vgg11(pretrained=False, num_classes=output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.channel_adaption(x)
        x = self.scores(x)
        x = self.softmax(x)
        return x


class ResNet18(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(ResNet18, self).__init__()
        self.scores = resnet18(pretrained=False, num_classes=output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.scores(x)
        x = self.softmax(x)
        return x


class TinyCNNContinous(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, scale: float = 1.5):
        super().__init__()

        self._scale = scale
        conv1_size = 16
        conv2_size = 32
        conv3_size = 64
        fc1_size = 32
        number_of_pooling_layers = 1
        self._fc1_input_size = int(
            conv3_size * int(rows / (2 * number_of_pooling_layers)) * int(cols / (2 * number_of_pooling_layers)))

        self.conv1 = torch.nn.Conv2d(input_channels, conv1_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, output_size)
        print(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        means = torch.tanh(x[0, :2]) * self._scale
        stds = x[0, 2:]
        stds = F.relu(stds) + 0.1
        x = torch.cat([means, stds]).unsqueeze(0)
        del means, stds
        return x


class TinyCNNMixed(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_sizes: Tuple[int], rows: int, cols: int, scale: float = 1.5):
        super().__init__()

        self._scale = scale
        conv1_size = 16
        conv2_size = 32
        conv3_size = 64
        fc1_size = 32
        number_of_pooling_layers = 1
        self._fc1_input_size = int(
            conv3_size * int(rows / (2 * number_of_pooling_layers)) * int(cols / (2 * number_of_pooling_layers)))

        self.conv1 = torch.nn.Conv2d(input_channels, conv1_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, fc1_size)
        self.fc_out1 = torch.nn.Linear(fc1_size, output_sizes[0])
        self.fc_out2 = torch.nn.Linear(fc1_size, output_sizes[1])
        self.softmax = nn.Softmax(dim=-1)
        print(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        mean_std = self.fc_out1(x)
        probs = self.fc_out2(x)
        probs = self.softmax(probs).squeeze(0)
        mean = (pi / 2) * torch.tanh(mean_std[0, 0])
        mean = mean.unsqueeze(0)
        std = torch.exp(x[0, 1]).unsqueeze(0)
        mean_std = torch.cat([mean, std])
        x = torch.cat([mean_std, probs]).unsqueeze(0)
        del probs, mean, std, mean_std
        return x


class PolicyCNN(nn.Module):

    def __init__(self, shared, fc_in, nbr_actions):
        super().__init__()
        self.shared = shared
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, nbr_actions)

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.shared(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
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

        # Softmax output
        self.softmax = nn.Softmax(dim=-1)

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

        # mean & standard deviation
        means = torch.tanh(x[0, :2]) * self._scale
        stds = x[0, 2:]
        stds = F.relu(stds) + 0.1
        x = torch.cat([means, stds]).unsqueeze(0)
        del means, stds

        return x


class ConvNet(torch.nn.Module):

    def __init__(self, input_channels: int, p: float = 0.1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=p)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = standardize_tensor_nn(x)
        x = self.activation(self.conv1(x))
        x = self.dropout2d(x)
        x = self.activation(self.conv2(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.dropout2d(x)
        x = self.activation(self.conv4(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = self.activation(self.conv5(x))
        x = self.dropout2d(x)
        return x


class SmallerConvNet(torch.nn.Module):

    def __init__(self, input_channels: int, p: float = 0.1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=p)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = standardize_tensor(x)
        x = self.activation(self.conv1(x))
        x = self.dropout2d(x)
        x = self.activation(self.conv2(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.dropout2d(x)
        return x


class DropoutCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.75):
        super().__init__()

        self._scale = torch.tensor(2, requires_grad=False)
        self._fc1_input_size = 128
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        # conv3_size = conv_size(pool1_size, 3, 0, 1)
        # self._fc1_input_size *= conv3_size ** 2
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self._conv_net = ConvNet(input_channels, dropout_probability)
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 512)  # last: 512
        self.fc2 = torch.nn.Linear(512, 512)
        self.param_head = torch.nn.Linear(512, 2)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()

        self.log_stds = torch.nn.Parameter(torch.log(1. * torch.ones(2)))
        self.log_stds.requires_grad = False

    def forward(self, x):
        x, v, a = x[0], x[1].unsqueeze(0).unsqueeze(0), x[2].unsqueeze(0).unsqueeze(0)
        x = self._conv_net(x)
        x = x.view(-1, self._fc1_input_size)
        x = torch.cat([x, v, a], dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.activation(self.fc2(x))
        x = self.dropout1d(x)

        # params = self.param_head(x)
        # mean_v = torch.clamp(params[:, :1], -1.75, 1.75)
        # mean_a = torch.clamp(params[:, 1:], -pi/4., pi/4.)

        params = torch.tanh(0.5 * self.param_head(x))
        mean_v = params[:, :1] * 1.05 + 0.75
        mean_a = params[:, 1:] * pi / 4.

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]  # log_stds[:, 0], log_stds[:, 1]

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, 2)
        for i in range(n):
            xs[i] = torch.tensor(self(x))
        return xs.std(dim=0) ** 2


class SmallerDropoutCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.75):
        super().__init__()

        self._scale = torch.tensor(2, requires_grad=False)
        self._fc1_input_size = 128
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        # conv3_size = conv_size(pool1_size, 3, 0, 1)
        # self._fc1_input_size *= conv3_size ** 2
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self._conv_net = ConvNet(input_channels, dropout_probability)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.param_head = torch.nn.Linear(512, 2)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()

        self.log_stds = torch.nn.Parameter(torch.log(1. * torch.ones(2)))
        self.log_stds.requires_grad = False

    def forward(self, x):
        x, v, a = x[0], x[1].unsqueeze(0).unsqueeze(0), x[2].unsqueeze(0).unsqueeze(0)
        x = self._conv_net(x)
        x = x.view(-1, self._fc1_input_size)
        # x = torch.cat([x, v, a], dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)

        params = torch.tanh(self.param_head(x))
        mean_v = (params[:, :1] + 1.) / 2. * 1.2
        mean_a = params[:, 1:] * pi

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]  # log_stds[:, 0], log_stds[:, 1]

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, 2)
        for i in range(n):
            xs[i] = torch.tensor(self(x))
        return xs.std(dim=0) ** 2


class DropoutCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.75):
        super().__init__()

        self._scale = torch.tensor(2, requires_grad=False)
        self._fc1_input_size = 128
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        # conv3_size = conv_size(pool1_size, 3, 0, 1)
        # self._fc1_input_size *= conv3_size ** 2
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self._conv_net = ConvNet(input_channels, dropout_probability)
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 512)  # last: 512
        self.fc2 = torch.nn.Linear(512, 512)
        self.param_head = torch.nn.Linear(512, 2)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()

        self.log_stds = torch.nn.Parameter(torch.log(1. * torch.ones(2)))
        self.log_stds.requires_grad = False

    def forward(self, x):
        x, v, a = x[0], x[1].unsqueeze(0).unsqueeze(0), x[2].unsqueeze(0).unsqueeze(0)
        x = self._conv_net(x)
        x = x.view(-1, self._fc1_input_size)
        x = torch.cat([x, v, a], dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.activation(self.fc2(x))
        x = self.dropout1d(x)

        # params = self.param_head(x)
        # mean_v = torch.clamp(params[:, :1], -1.75, 1.75)
        # mean_a = torch.clamp(params[:, 1:], -pi/4., pi/4.)

        params = torch.tanh(0.5 * self.param_head(x))
        mean_v = params[:, :1] * 1.05 + 0.75
        mean_a = params[:, 1:] * pi / 4.

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]  # log_stds[:, 0], log_stds[:, 1]

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, 2)
        for i in range(n):
            xs[i] = torch.tensor(self(x))
        return xs.std(dim=0) ** 2


class PolicyCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.8):
        super().__init__()
        self.actor = DropoutCNN(input_channels, output_size, rows, cols, dropout_probability, scale)


class SimplerPolicyCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.8):
        super().__init__()
        self.actor = SmallerDropoutCNN(input_channels, output_size, rows, cols, dropout_probability, scale)


class DropoutCNNDiscreteActions(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.75):
        super().__init__()

        self._scale = torch.tensor(2, requires_grad=False)
        self._fc1_input_size = 128
        pool1_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        # conv3_size = conv_size(pool1_size, 3, 0, 1)
        # self._fc1_input_size *= conv3_size ** 2
        pool2_size = 0.5 * conv_size(conv_size(pool1_size, 3, 0, 1), 3, 0, 1)
        conv5_size = conv_size(pool2_size, 3, 0, 1)
        self._fc1_input_size *= conv5_size ** 2

        self._conv_net = ConvNet(input_channels, dropout_probability)
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 512)  # last: 512
        self.fc2 = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, output_size)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.activation = torch.nn.ReLU()  # torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x, v, a = x[0], x[1].unsqueeze(0).unsqueeze(0), x[2].unsqueeze(0).unsqueeze(0)
        x = self._conv_net(x)
        x = x.view(-1, self._fc1_input_size)
        x = torch.cat([x, v, a], dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.activation(self.fc2(x))
        x = self.dropout1d(x)

        x = self.out(x)
        x = self.softmax(x)
        return x

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, 2)
        for i in range(n):
            xs[i] = torch.tensor(self(x))
        return xs.std(dim=0) ** 2


class PolicyCNNDiscreteActions(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.8):
        super().__init__()
        self.actor = DropoutCNNDiscreteActions(input_channels, output_size, rows, cols, dropout_probability, scale)


class FlexibleMLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.fcs = torch.nn.ModuleList()
        if len(n_neurons) == 0:
            self.fcs += [torch.nn.Linear(self._input_size, output_size)]
        else:
            self.fcs += [torch.nn.Linear(self._input_size, n_neurons[0])]
            for l in range(n_hidden - 1):
                self.fcs += [torch.nn.Linear(n_neurons[l], n_neurons[l + 1])]
            self.fcs += [torch.nn.Linear(n_neurons[-1], self._output_size)]

        self.activation = activation
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.log_stds = torch.nn.Parameter(torch.log(1. * torch.ones(2)))
        self.log_stds.requires_grad = False

    def forward(self, x):
        x = self.normalize_kinematics(x)
        for linear in self.fcs[:-1]:
            x = self.activation(linear(x))
            x = self.dropout1d(x)
        params = torch.tanh(self.fcs[-1](x)).unsqueeze(0)
        mean_v = params[:, :1]
        mean_a = params[:, 1:]

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]  # log_stds[:, 0], log_stds[:, 1]

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2

    def normalize_kinematics(self, x,
                             min_dist=0., max_dist=11.4025 ** 0.5,
                             min_angle=0., max_angle=2 * pi,
                             min_velocity=-0.3, max_velocity=1.8):
        n_agents = self._n_agents
        dists = [2] + [i for i in range(4, 3 * (n_agents - 1) + 2, 3)] + [i for i in range(-8, 0, 2)]
        angles = [1] + [i for i in range(5, 3 * (n_agents - 1) + 3, 3)] + [i for i in range(-7, 0, 2)]
        velocities = [0] + [i for i in range(3, 3 * (n_agents - 1) + 1, 3)]

        x[:, dists] = (x[:, dists] - min_dist) / (max_dist - min_dist)
        x[:, angles] = (x[:, angles] - min_angle) / (max_angle - min_angle)
        x[:, velocities] = (x[:, velocities] - min_velocity) / (max_velocity - min_velocity)

        return x


class MLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()
        self.actor = FlexibleMLP(input_size, output_size, n_hidden, n_neurons, activation, dropout_probability)


class Polynomial(torch.nn.Module):

    def __init__(self, n_agents, n_obstacles, output_size):
        super().__init__()
        self.actor = PolynomialPolicy(n_agents, n_obstacles, output_size)


class PolynomialPolicy(torch.nn.Module):

    def __init__(self, n_agents, n_obstacles, output_size):
        super().__init__()
        self.n_ag = n_agents
        self.n_ob = n_obstacles
        self.dim_basis = 5 + (self.n_ag - 1) * 3 + self.n_ob * 3

        self.l = nn.Linear(self.dim_basis, output_size)

    def basis_transform(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        phi = torch.zeros([x.shape[0], self.dim_basis])
        phi[:, :2] = x[:, :2]
        phi[:, 2:4] = x[:, 2:4]
        phi[:, 4] = x[:, 2] * x[:, 3]

        for i in range(1, self.n_ag):
            phi[:, 5 + 3 * i] = x[:, 4 + 2 * i]
            phi[:, 5 + 3 * i + 1] = x[:, 4 + 2 * i + 1]
            phi[:, 5 + 3 * i + 2] = x[:, 4 + 2 * i] * x[:, 4 + 2 * i + 1]

        for o in range(self.n_ob):
            phi[:, 5 + 3 * (self.n_ag - 1) + 3 * o] = x[:, 4 + 2 * (self.n_ag - 1) + 2 * o]
            phi[:, 5 + 3 * (self.n_ag - 1) + 3 * o + 1] = x[:, 4 + 2 * (self.n_ag - 1) + 2 * o + 1]
            phi[:, 5 + 3 * (self.n_ag - 1) + 3 * o + 2] = x[:, 4 + 2 * (self.n_ag - 1) + 2 * o] * x[:, 4 + 2 * (
                        self.n_ag - 1) + 2 * o + 1]

        return phi

    def forward(self, x):
        mean = self.l(self.basis_transform(x))
        return mean[:, 0].unsqueeze(0), mean[:, 1].unsqueeze(0), 0., 0.


class FlexibleMLPDiscreteActions(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.fcs = torch.nn.ModuleList()
        if len(n_neurons) == 0:
            self.fcs += [torch.nn.Linear(self._input_size, output_size)]
        else:
            self.fcs += [torch.nn.Linear(self._input_size, n_neurons[0])]
            for l in range(n_hidden - 1):
                self.fcs += [torch.nn.Linear(n_neurons[l], n_neurons[l + 1])]
            self.fcs += [torch.nn.Linear(n_neurons[-1], self._output_size)]

        self.activation = activation
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.log_stds = torch.nn.Parameter(torch.log(1. * torch.ones(2)))
        self.log_stds.requires_grad = False

        self.softmax = torch.nn.Softmax(dim=-1)

        self._n_agents = int((input_size - 6 * 2) / 3)

    def forward(self, x):
        #x = self.normalize_kinematics(x)
        for linear in self.fcs:
            x = self.activation(linear(x))
            x = self.dropout1d(x)
        out = self.softmax(x)
        return out

    def var(self, x, n: int = 100):
        xs = torch.zeros(n, self._output_size)
        for i in range(n):
            xs[i] = self(x)
        return xs.std(dim=0) ** 2


    def normalize_kinematics(self, x,
                             min_dist=-0.15, # -soft_person_radius
                             max_dist=(1.55 ** 2 + 3 ** 2) ** 0.5, # diagonal size, can be exceeded
                             min_angle=0., max_angle=2 * pi,
                             min_velocity=-0.3, max_velocity=1.8): # backward, forward
        n_agents = self._n_agents
        dists = [2] + [i for i in range(4, 3 * n_agents - 1, 3)] + [i for i in range(-12, 0, 2)]
        angles = [1] + [i for i in range(5, 3 * n_agents, 3)] + [i for i in range(-11, 0, 2)]
        velocities = [0] + [i for i in range(3, 3 * n_agents - 2, 3)]

        x[:, dists] = (x[:, dists] - min_dist) / (max_dist - min_dist)
        x[:, angles] = (x[:, angles] - min_angle) / (max_angle - min_angle)
        x[:, velocities] = (x[:, velocities] - min_velocity) / (max_velocity - min_velocity)

        return x


class MLPDiscreteActions(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()
        self.actor = FlexibleMLPDiscreteActions(input_size, output_size, n_hidden, n_neurons, activation, dropout_probability)