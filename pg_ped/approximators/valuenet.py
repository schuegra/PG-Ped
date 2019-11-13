import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18, alexnet

from pg_ped.utils import standardize_tensor, normalize_tensor


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

        print(self)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self._fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super(SimpleCNN, self).__init__()
        self._fc1_input_size = int(512 * int(rows / 4) * int(cols / 4))

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(self._fc1_input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

    def forward(self, x):
        x = normalize_tensor(standardize_tensor(x))
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


class AlexNet(torch.nn.Module):

    # Our batch shape for input x is (backward_view, rows, cols)

    def __init__(self, input_channels: int, output_size: int, rows: int, cols: int):
        super().__init__()
        self.channel_adaption = torch.nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        self.scores = alexnet(pretrained=False, num_classes=output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.channel_adaption(x)
        x = self.scores(x)
        x = self.softmax(x)
        return x




class ValueCNN(nn.Module):

    def __init__(self, shared, fc_in):
        super().__init__()
        self.shared = shared
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x
#
# def alexnet(pretrained=False, progress=True, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = AlexNet(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#
#     return model
