from math import pi as math_pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18

from pg_ped.visualization.visualize_cnn import vis_feature_maps, vis_fc
from pg_ped.utils import standardize_tensor


class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super().__init__()
        self._fc1_input_size = int(512 * int(rows / 4.) * int(cols / 4.))

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))

        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        return x


def conv_size(W, F, P, S):
    return int((W - F + 2 * P) * 1. / S) + 1


class MiniCNN(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super().__init__()
        self._fc1_input_size = int(32 * round(rows / 1) * round(cols / 1))

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
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


class DropoutCNN(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super(DropoutCNN, self).__init__()

        self._scale = scale

        self._fc1_input_size = 128
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.means_stds = torch.nn.Linear(512, 4)
        # self.mean_v =
        # self.mean_a = torch.nn.Linear(256, 1)
        # self.std_v = torch.nn.Linear(256, 1)
        # self.std_a = torch.nn.Linear(256, 1)
        self.value_head = torch.nn.Linear(512, 1)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2d(x)

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)

        # mean & standard deviation
        means_stds = self.means_stds(x)
        mean_v = torch.sigmoid(means_stds[:, 0:1]) * self._scale
        mean_a = torch.sigmoid(means_stds[:, 1:2]) * 2 * math_pi
        std_v = F.relu(means_stds[:, 2:3]) + 1e-8
        std_a = F.relu(means_stds[:, 3:4]) + 1e-8

        # Value
        value = self.value_head(x)

        return mean_v, mean_a, std_v, std_a, value


class DDPGNet(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self._scale = scale

        self._fc1_input_size = 128
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.action_head = torch.nn.Linear(512, 2)
        self.value_head = torch.nn.Linear(512 + 2, 1)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

        self.loss = torch.nn.SmoothL1Loss()  # Behaves linear instead of quadratic (MSE) at some distance from the minimum

    def forward(self, x):
        # Preprocess
        x = standardize_tensor(x)

        # Feature Extraction
        x = F.relu(self.conv1(x))
        # x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        # x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # x = self.dropout2d(x)

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)

        # action & value
        action = self.action_head(x)
        action[:, 0] = torch.sigmoid(action[:, 0]) * self._scale
        action[:, 1] = torch.sigmoid(action[:, 1]) * 2 * math_pi
        value = self.value_head(torch.cat([x, action], dim=1))

        return action, value


class Actor(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self._scale = scale

        self._fc1_input_size = 128
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 512)
        self.action_head = torch.nn.Linear(512, 2)
        self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

    def forward(self, x):
        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2d(x)

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)

        # action
        action = self.action_head(x)
        action[:, 0] = torch.tanh(action[:, 0]) * self._scale
        action[:, 1] = torch.sigmoid(action[:, 1]) * math_pi

        return action


class Critic(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self._scale = scale

        self._fc1_input_size = 64
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        self._fc1_input_size *= conv3_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 256)
        self.value_head = torch.nn.Linear(256, 1)
        # self.dropout2d = torch.nn.Dropout2d(p=dropout_probability)
        # self.dropout1d = torch.nn.Dropout(p=dropout_probability)

    def forward(self, x):

        x, action = x

        # Feature Extraction
        x = F.relu(self.conv1(x))
        # x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        # x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # x = self.dropout2d(x)

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(torch.cat([x, action], dim=1)))
        # x = self.dropout1d(x)

        # value
        value = self.value_head(x)

        return value

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SmallActor(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self._scale = scale

        self._fc1_input_size = 32
        conv2_size = conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        self._fc1_input_size *= conv2_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.action_head = torch.nn.Linear(128, 2)

    def forward(self, x):
        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))

        # action
        action = self.action_head(x)
        action[:, 0] = torch.tanh(action[:, 0]) * self._scale
        action[:, 1] = torch.sigmoid(action[:, 1]) * math_pi

        return action


class SmallCritic(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self._scale = scale

        self._fc1_input_size = 32
        conv2_size = conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        self._fc1_input_size *= conv2_size ** 2

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 128)
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x):

        x, action = x

        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Scores
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(torch.cat([x, action], dim=1)))

        # value
        value = self.value_head(x)

        return value

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class BNCritic(torch.nn.Module):

    def __init__(self, conv_net: torch.nn.Module, fc1_input_size: int, scale: float = 1.5):
        super().__init__()

        self._scale = scale
        self._fc1_input_size = fc1_input_size

        self._conv_net = conv_net
        self.fc1 = torch.nn.Linear(self._fc1_input_size + 2, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x):
        x, action = x
        x = self._conv_net(x)
        x = F.relu(self.bn1(self.fc1(torch.cat([x.view(-1, self._fc1_input_size), action], dim=1))))
        x = F.relu(self.bn2(self.fc2(x)))
        value = self.value_head(x)

        return value


class BNActor(torch.nn.Module):

    def __init__(self, conv_net: torch.nn.Module, fc1_input_size: int, scale: float = 1.5):
        super().__init__()

        self._scale = scale
        self._fc1_input_size = fc1_input_size

        self._conv_net = conv_net
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.action_head = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self._conv_net(x)
        x = F.relu(self.bn1(self.fc1(x.view(-1, self._fc1_input_size))))
        x = F.relu(self.bn2(self.fc2(x)))
        action = self.action_head(x)
        action[:, 0] = torch.tanh(action[:, 0]) * self._scale
        action[:, 1] = torch.sigmoid(action[:, 1]) * math_pi

        return action


class ACNet(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self.actor = Actor(input_channels, output_size, rows, cols, dropout_probability, scale)
        self.critic = Critic(input_channels, output_size, rows, cols, dropout_probability, scale)


class SmallACNet(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        self.actor = SmallActor(input_channels, output_size, rows, cols, dropout_probability, scale)
        self.critic = SmallCritic(input_channels, output_size, rows, cols, dropout_probability, scale)


class ConvNet(torch.nn.Module):

    def __init__(self, input_channels: int, p: float = 0.1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=p)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout2d(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2d(x)
        return x


class ConvNetNoBN(torch.nn.Module):

    def __init__(self, input_channels: int, p: float = 0.1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=p)

    def forward(self, x):
        x = standardize_tensor(x)
        x = F.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout2d(x)
        return x


class BNSharedACNet(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 scale: float = 1.5):
        super().__init__()

        fc1_input_size = 64
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        fc1_input_size *= conv3_size ** 2

        conv_net = ConvNet(input_channels)
        self.actor = BNActor(conv_net, fc1_input_size, scale)
        self.critic = BNCritic(conv_net, fc1_input_size, scale)


class A2CCritic(torch.nn.Module):

    def __init__(self, conv_net: torch.nn.Module, fc1_input_size: int, p: float = 0.1):
        super().__init__()

        self._fc1_input_size = fc1_input_size

        self._conv_net = conv_net
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 256)
        # self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        self.value_head = torch.nn.Linear(128, 1)
        self.dropout1d = torch.nn.Dropout(p=p)

    def forward(self, x):
        x = self._conv_net(x)
        x = F.relu(self.fc1(x.view(-1, self._fc1_input_size)))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        value = self.value_head(x)

        return value


class A2CActor(torch.nn.Module):

    def __init__(self, conv_net: torch.nn.Module, fc1_input_size: int, v_scale: float = 1.5, a_scale: float = math_pi,
                 p: float = 0.1):
        super().__init__()

        self._scale = torch.tensor(2, requires_grad=False)
        self._fc1_input_size = fc1_input_size

        self._conv_net = conv_net
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 256)
        # self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        self.param_head = torch.nn.Linear(128, 2)
        self.dropout1d = torch.nn.Dropout(p=p)

        self.log_stds = torch.nn.Parameter(torch.Tensor(2, ))

    def forward(self, x):
        x = self._conv_net(x)
        x = F.relu(self.fc1(x.view(-1, self._fc1_input_size)))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1d(x)
        params = torch.tanh(self.param_head(x)) * self._scale
        mean_v = params[:, :1]
        mean_a = params[:, 1:]

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]


class A2CNet(torch.nn.Module):

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int, dropout_probability: float,
                 v_scale: float = 1.5, a_scale: float = math_pi):
        super().__init__()

        fc1_input_size = 64
        pool_size = 0.5 * conv_size(conv_size(rows, 3, 0, 1), 3, 0, 1)
        conv3_size = conv_size(pool_size, 3, 0, 1)
        fc1_input_size *= conv3_size ** 2

        conv_net_actor = ConvNetNoBN(input_channels, dropout_probability)
        conv_net_critic = ConvNetNoBN(input_channels, dropout_probability)
        self.actor = A2CActor(conv_net_actor, fc1_input_size, v_scale, a_scale, dropout_probability)
        self.critic = A2CCritic(conv_net_critic, fc1_input_size, dropout_probability)


class MLPCritic(torch.nn.Module):

    def __init__(self, input_size: int, dropout_probability: float):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.activation = torch.nn.ReLU()
        self.dropout1d = torch.nn.Dropout(p=dropout_probability)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)
        return x


class MLPActor(torch.nn.Module):

    def __init__(self, input_size: int, v_scale: float = 1.75, a_scale: float = math_pi / 4., p: float = 0.1):
        super().__init__()

        self._v_scale = v_scale
        self._a_scale = a_scale

        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.activation = torch.nn.ReLU()
        self.dropout1d = torch.nn.Dropout(p=p)

        self.log_stds = torch.nn.Parameter(-10 * torch.ones(2, ))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1d(x)
        params = torch.tanh(self.fc2(x))
        mean_v = self._v_scale * params[:, :1]
        mean_a = self._a_scale * params[:, 1:]

        return mean_v, mean_a, self.log_stds[0], self.log_stds[1]


class A2CMLP(torch.nn.Module):

    def __init__(self, input_size: int, dropout_probability: float):
        super().__init__()
        self.actor = MLPActor(input_size, p=dropout_probability)
        self.critic = MLPCritic(input_size, dropout_probability)


class MLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float, forward):
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

        self._forward = forward

    def forward(self, *args):
        return self._forward(*args)


class DDPGMLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 n_hidden: int, n_neurons: List[int],
                 activation: torch.nn.Module, dropout_probability: float):
        super().__init__()

        self.actor = MLP(input_size, output_size, n_hidden, n_neurons, activation, dropout_probability,
                         self._actor_forward)
        self.critic = MLP(input_size + 2, 1, n_hidden, n_neurons, activation, dropout_probability,
                          self._critic_forward)

    def _actor_forward(self, x):
        for linear in self.fcs[:-1]:
            x = self.activation(linear(x))
            x = self.dropout1d(x)
        action = torch.tanh(self.fcs[-1](x)).unsqueeze(0)
        delta_v = params[:, :1]
        delta_a = params[:, 1:]
        return delta_v, delta_a

    def _critic_forward(self, x, action):
        x = torch.cat([x, action])
        for linear in self.fcs[:-1]:
            x = self.activation(linear(x))
            x = self.dropout1d(x)
        value = self.fcs[-1](x).unsqueeze(0)
        return value
