import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from PIL import Image
from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
from depth.ops import resize
#from depth.models.losses import sigloss

class DepthTFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
             **model_cfg.LOSS.ARGS
         )
        #self.depth_loss = SigLoss(loss_weight=10)
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        
        image_id = batch_dict["frame_id"]
        
        depth = ddn_result["pred_depths"]
        
        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["depth"] = depth
        return batch_dict

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        #loss = self.depth_loss(**self.forw/cmkd-mono-Vmam/pcdet/models/backbones_3d/vfe/image_vfe_modules/ffn/depth_ffn.py", line 150ard_ret_dict)#(self.forward_ret_dict["depth_logits"], self.forward_ret_dict["depth_maps"]) * 0.25
        return loss, tb_dict
