import torch
import torch.nn as nn


from .balancer import Balancer
from pcdet.utils import transform_utils

try:
    from kornia.losses.focal import FocalLoss
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_weight,
                 bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}

        # Bin depth map to create target / depth_maps 2,94,311 이거는 확률값같은 로짓?
        depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)
        loss = self.loss_func(depth_logits, depth_target)
        #gt_one_hot_value = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True) # soft_one_hot_gt
        #mask =  self.create_depth_mask(gt_one_hot_value)
        #extend_gt_one_hot_value = gt_one_hot_value * mask
        
        # Compute loss 2,151,94,311 / 2.94.311 depth_target은 150이 최대
        #loss = self.focal_loss(frustum_features_prob, extend_gt_one_hot_value[:, :-1, :, :]) 

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)

        # Final loss
        loss *= self.weight
        tb_dict.update({"ddn_loss": loss.item()})

        return loss, tb_dict


    def create_depth_mask(self, gt_one_hot_value, l=1):
        """
        Create depth mask M based on the given gt_one_hot_value (Ψ) and threshold l.
        
        Args:
            gt_one_hot_value (torch.Tensor): One-hot encoded depth values, shape (B, D, H, W)
            l (int): Threshold for depth range extension
        
        Returns:
            mask (torch.Tensor): Depth mask with the same shape as gt_one_hot_value
        """
        # Get the depth bin indices where gt_one_hot_value is 1 (i.e., z' values)
        depth_indices = torch.argmax(gt_one_hot_value, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Create a tensor representing bin indices (D) for comparison
        bin_indices = torch.arange(gt_one_hot_value.shape[1], device=gt_one_hot_value.device).view(1, -1, 1, 1)  # Shape: (1, D, 1, 1)
        
        # Compute mask condition |d - z'| ≤ l
        mask = (torch.abs(bin_indices - depth_indices) <= l).float()  # Shape: (B, D, H, W)
        
        return mask
    