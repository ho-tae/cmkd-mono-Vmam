import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from PIL import Image
from . import ddn, ddn_loss 
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
from torchvision.transforms.functional import gaussian_blur


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
        
        self.fru3D = nn.Sequential(
                                nn.Conv3d(32, 32, 3, 1, 1),
                                nn.GroupNorm(2, 32),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(32, 32, 3, 1, 1),
                                nn.GroupNorm(2, 32),
                                nn.ReLU(inplace=True)
        )
        self.fru3D_prob = nn.Sequential(nn.Conv3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, 3, 1, 1))
        
        
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
        
        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)
        
        
        # Create image feature plane-sweep volume
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                         depth_logits=depth_logits)  # 강화된 depth 맵 사용

        frustum_features = self.fru3D(frustum_features)

        frustum_features_prob = self.fru3D_prob(frustum_features)
        frustum_features_prob = torch.clamp(torch.sigmoid(frustum_features_prob), 1e-5, 1 - 1e-5)
        
        batch_dict["frustum_features_prob"] = frustum_features_prob
        batch_dict["frustum_features"] = image_features.unsqueeze(2) * frustum_features_prob
    
        
        #batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits
            
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (B, C, H, W)
            depth_logits: (B, D+1, H, W)
        Returns:
            frustum_features: (B, C, D, H, W)
        """
        
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]
        
        # Compute initial frustum features
        frustum_features = depth_probs * image_features
        enhanced_frustum_features = frustum_features + depth_probs
        return enhanced_frustum_features

    

    def create_depth_map_img(self, depth_probs, image_id, save_path="./caddn_vmam_skip_connec"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        channel_dim = 1
        depth_dim = 2
        
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

    def create_frustum_feature_img(self, frustum_features, image_id, save_path="./caddn_frustum_features_vis"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        batch_size, channels, depth_bins, height, width = frustum_features.shape

        # Depth Bin 설정 (0~1 사이 균등 분포)
        depth_bins = torch.linspace(0, 1, depth_bins).to(frustum_features.device).view(1, 1, -1, 1, 1)  # (1, 1, D, 1, 1)

        # Frustum Feature Map 생성 (Expected Depth Feature 계산)
        frustum_map = torch.sum(frustum_features * depth_bins, dim=2)  # (B, C, D, H, W) → (B, C, H, W)

        # 채널 차원 평균 (B, C, H, W) → (B, H, W)  
        frustum_map = frustum_map.mean(dim=1)  

        # 저장
        for i in range(batch_size):
            file_path = os.path.join(save_path, f"frustum_map_{image_id[i]}.png")

            # Tensor → NumPy 변환
            frustum_map_np = frustum_map[i].cpu().detach().numpy()

            # Normalize for saving as an 8-bit image
            frustum_map_norm = (frustum_map_np - frustum_map_np.min()) / (frustum_map_np.max() - frustum_map_np.min()) * 255
            frustum_map_image = Image.fromarray(frustum_map_norm.astype(np.uint8))
            frustum_map_image.save(file_path)


    def bin_ray_depths(self, depth_map, mode, depth_min, depth_max, num_bins, target=False):
        """a
        Converts depth map into bin indices
        Args:
            depth_map: (H, W), Depth Map
            mode: string, Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min: float, Minimum depth value
            depth_max: float, Maximum depth value
            num_bins: int, Number of depth bins
            target: bool, Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices: (H, W), Depth bin indices
        """
        import math
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                        (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        B, H, W = indices.shape
        gt_one_hot = torch.zeros((B, num_bins + 1, H, W), device=indices.device)
        gt_one_hot = gt_one_hot.scatter(1, indices.unsqueeze(1), 1)  # One-hot encoding

        return gt_one_hot
    

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        
        #loss, tb_dict = self.occupancy_loss(**self.forward_ret_dict)
        #loss += loss2
        #tb_dict.update(tb_dict2)
        
        return loss, tb_dict
    
    def occupancy_loss(self, gt_one_hot, frustum_features_prob, **kwargs):
        
        mask = self.create_depth_mask(gt_one_hot)
        
        extend_gt_one_hot = gt_one_hot * mask
        
        cls_loss = self.focal_loss_2(frustum_features_prob, extend_gt_one_hot[:, :-1, ...])
        
        #cls_loss = self.focal_loss(frustum_features_prob, depth_target[:, :-1, ...])

        tb_dict = {"occupancy_loss": cls_loss.item()}
        return cls_loss, tb_dict
    
    
    # def create_depth_mask(self, gt_one_hot_value, l=1):
    #     """
    #     Create depth mask M based on the given gt_one_hot_value (Ψ) and threshold l.
        
    #     Args:
    #         gt_one_hot_value (torch.Tensor): One-hot encoded depth values, shape (B, D, H, W)
    #         l (int): Threshold for depth range extension
        
    #     Returns:
    #         mask (torch.Tensor): Depth mask with the same shape as gt_one_hot_value
    #     """
    #     # Get the depth bin indices where gt_one_hot_value is 1 (i.e., z' values)
    #     depth_indices = torch.argmax(gt_one_hot_value, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
    #     # Create a tensor representing bin indices (D) for comparison
    #     bin_indices = torch.arange(gt_one_hot_value.shape[1], device=gt_one_hot_value.device).view(1, -1, 1, 1)  # Shape: (1, D, 1, 1)
        
    #     # Compute mask condition |d - z'| ≤ l
    #     mask = (torch.abs(bin_indices - depth_indices) <= l).float()  # Shape: (B, D, H, W)
        
    #     return mask
    
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
        
        # Compute mask condition |d - z'| > l or d = z'
        mask = (torch.abs(bin_indices - depth_indices) > l) | (bin_indices == depth_indices)  # Shape: (B, D, H, W)
        
        # Convert the mask from boolean to float
        mask = mask.float()  # Shape: (B, D, H, W)
        
        return mask
    
    def focal_loss_2(self, input, target, alpha=0.25, gamma=2.):
        '''
        Args:
            input:  prediction, 'batch x c x h x w'
            target:  ground truth, 'batch x c x h x w'
            alpha: hyper param, default in 0.25
            gamma: hyper param, default in 2.0
        Reference: Focal Loss for Dense Object Detection, ICCV'17
        '''
        
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        loss = 0

        pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
        neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss.mean()