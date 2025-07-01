# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from pcdet.utils import transform_utils
from .balancer import Balancer
from depth.ops import resize
from kornia.losses.focal import FocalLoss

class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100,
                 weight=0.25
                 ):
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.weight = weight
        self.align_corners = True
        self.eps = 0.001 # avoid grad explode
        
        self.loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.balancer = Balancer(downsample_factor=4,
                                 fg_weight=13,
                                 bg_weight=1)
        self.disc_cfg=dict(mode='LID',
                           num_bins=150,
                           depth_min=2.0,
                           depth_max=46.8)
        
        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth, depth_maps): # (self, depth_logits, gt_boxes2d, depth, depth_maps):
        """Forward function."""
        
        tb_dict = {}
        
        #depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        depth_maps_2 = depth_maps.unsqueeze(1)
        depth = resize(
            input=depth,
            size=depth_maps_2.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        
        loss_depth = self.loss_weight * self.sigloss(depth, depth_maps_2)
        
        #loss = self.loss_func(depth_logits, depth_target)
        
        # # Compute foreground/background balancing
        #loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        
        #loss_depth *= self.weight
        
        #loss_depth = loss_depth + loss 
        
        tb_dict.update({"ddn_loss": loss_depth.item()})
        
        return loss_depth, tb_dict
