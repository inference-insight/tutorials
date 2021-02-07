import torch.nn as nn
import torch.nn.functional as F

class Conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        dilation=1
    ):
        super(Conv2DBatchNormRelu, self).__init__()

        conv_layer = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.conv_bn_layer = nn.Sequential(conv_layer, nn.BatchNorm2d(int(out_channels)))
        
    def forward(self, x):
        y = F.relu(self.conv_bn_layer(x))
        return y


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()

        # convolutional layers for feature extraction
        self.conv1 = Conv2DBatchNormRelu(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2DBatchNormRelu(in_channels=16, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2DBatchNormRelu(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)

        # fully connected layers for classification
        self.fc1 = nn.Linear(64*3*3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # send the input through the layers
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        # flatten the features
        x = x.view(-1, 64*3*3)

        # apply fully-connected layers
        x = F.dropout(F.relu(self.fc1(x)), 0.2)
        x = F.dropout(F.relu(self.fc2(x)), 0.2)
        x = self.fc3(x)
        return x