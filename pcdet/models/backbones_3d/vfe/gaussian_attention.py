import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianRefinementBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, init_sigma=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.ones(1) * init_sigma)
        self.padding = kernel_size // 2

    def create_gaussian_kernel(self, device):
        k = self.kernel_size
        sigma = self.sigma.clamp(min=0.1)
        ax = torch.arange(-k // 2 + 1., k // 2 + 1., device=device)
        xx, yy, zz = torch.meshgrid([ax, ax, ax], indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k, k).repeat(self.in_channels, 1, 1, 1, 1)
        return kernel

    def forward(self, x):
        B, C, D, H, W = x.shape
        kernel = self.create_gaussian_kernel(x.device)
        refined = F.conv3d(x, kernel, padding=self.padding, groups=C)
        return x + refined


class GaussianOccupancyVFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.refine_block = GaussianRefinementBlock(in_channels)
        self.occupancy_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, voxel_feats):
        refined = self.refine_block(voxel_feats)
        occupancy_logits = self.occupancy_head(refined)
        occupancy_prob = torch.sigmoid(occupancy_logits)
        refined_voxel_feats = refined * occupancy_prob
        return occupancy_prob, refined_voxel_feats


