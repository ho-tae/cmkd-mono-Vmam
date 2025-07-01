import torch
import torch.nn as nn
import torch.nn.functional as F

# class DepthHead(nn.Module):
#     def __init__(self, in_channels=128, out_channels=151):  
#         super(DepthHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(256, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         depth_logits = self.conv2(x)  # (N, 151, H, W)
#         return depth_logits


class DepthHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=151):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)  # BatchNorm 추가
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)  # BatchNorm 추가
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # BatchNorm + ReLU 추가
        x = F.relu(self.bn2(self.conv2(x)))  # BatchNorm + ReLU 추가
        depth_logits = self.conv3(x)  # Raw logits 반환 (Softmax는 후처리)
        return depth_logits
