# model.py
# cnn models for emnist dataset

import warnings

warnings.filterwarnings("ignore")

import torch.nn as nn

warnings.filterwarnings("ignore")

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN,self).__init__()

        self.layer1 = self.conv_module(1, 16, IS_DROPOUT=False)
        self.layer2 = self.conv_module(16, 24, IS_DROPOUT=False)
        self.layer3 = self.conv_module(24, 32, IS_DROPOUT=False)
        self.layer4 = self.conv_module(32, 64, IS_DROPOUT=False)
        self.layer5 = self.conv_module(64, 128, IS_DROPOUT=False)
        self.fc_layer = self.global_avg_pool(128, 10, IS_DROPOUT=True)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc_layer(out)
        out = out.view(-1, 10)

        return out

    def conv_module(self,in_num, out_num, IS_DROPOUT=True):
        if IS_DROPOUT:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(kernel_size=3,stride=1)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1)
            )

    def global_avg_pool(self,in_num, out_num, IS_DROPOUT=True):
        if IS_DROPOUT:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.Dropout2d(p=0.5),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )