import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from PIL import Image
from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class DepthFFN(nn.Module):

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
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

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
        depth_logits = ddn_result["logits"]
        image_id = batch_dict["frame_id"]
        
        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        # Create image feature plane-sweep volume
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits,
                                                        image_id=image_id)
        batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits, image_id, depth_map=None):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range) ,각 로짓 값을 확률로 변환
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        if depth_map:
            create_depth_map_img(self, depth_probs, image_id)
        
        # Multiply to form image depth feature volume
        # 곱함으로써 픽셀별 깊이 확률을 이미지 특성에 반영하여 깊이 정보와 채널 특성을 결합하여 새로운 Frustum Feature Volume을 생성
        frustum_features = depth_probs * image_features 
        
        return frustum_features


    def create_depth_map_img(self, depth_probs, image_id, save_path="./depth_maps"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Depth Map 생성 (Expected Depth 계산)
        depth_bins = torch.linspace(0, 1, depth_probs.size(depth_dim)).to(depth_probs.device)  # Depth bin 설정 (예: 0~1 사이 균등 분포)
        depth_map = torch.sum(depth_probs * depth_bins.view(1, -1, 1, 1), dim=depth_dim)  # 각 픽셀의 기대 깊이값 계산

        # Depth Map 저장
        for i in range(depth_map.size(0)):
            file_path = os.path.join(save_path, f"depth_map_{image_id[i]}.png")
            
            # Squeeze to remove unnecessary dimensions (e.g., (1, 1, 311) -> (311,))
            depth_map_np = depth_map[i].cpu().detach().numpy().squeeze()

            # Normalize depth map for saving as an 8-bit image
            depth_map_norm = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min()) * 255
            depth_map_image = Image.fromarray(depth_map_norm.astype(np.uint8))
            depth_map_image.save(file_path)

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict
