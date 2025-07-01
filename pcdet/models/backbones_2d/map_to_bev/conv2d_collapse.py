import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from PIL import Image
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)

    def forward(self, batch_dict, bev_visual=False):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        
        if bev_visual:
            self.save_bev_image(batch_dict)
        
        return batch_dict


    def save_bev_image(self, batch_dict, batch_idx=0, channel_idx=None):
        bev_features = batch_dict["spatial_features"]  # (B, C, Y, X)

        # 배치에서 하나 선택
        bev_image = bev_features[batch_idx].cpu().detach().numpy()  # (C, Y, X)
        frame_id = batch_dict["frame_id"][0]
        save_path = (f"caddn_vmam_occupancy_bev_color_output_{frame_id}.png")
        # save_path_1 = (f"ours_bev_output_{frame_id}.png")
        # save_path_2 = (f"ours_bev_distribution_output_{frame_id}.png")
        
        # 특정 채널 선택 (없으면 평균)
        if channel_idx is not None:
            bev_image = bev_image[channel_idx]  # (Y, X)
        else:
            bev_image = bev_image.mean(axis=0)  # 모든 채널 평균
            
        cmap="viridis"
        
        # 시각화 (흰색 대신 컬러맵 적용)
        plt.figure(figsize=(8, 6))
        plt.imshow(bev_image, cmap=cmap, interpolation="nearest")
        plt.colorbar(label="Feature Value")  # 컬러바 추가
        plt.axis("off")  # 축 숨기기
        plt.title("BEV Feature Map")

        # Matplotlib 플롯을 PIL 이미지로 변환
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image.save(save_path)
        plt.close()
        
        # bev_values = bev_image.flatten()
        # bins=150
        
        # # 히스토그램 및 KDE 플롯 저장
        # plt.figure(figsize=(8, 6))
        # sns.histplot(bev_values, bins=bins, kde=True, stat="density", color="blue", alpha=0.6)

        # plt.xlabel("Feature Value")
        # plt.ylabel("Density")
        # plt.title("BEV Feature Distribution")
        # plt.grid(True)

        # plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        # plt.close()
        
        # # 정규화 (0~255로 변환)
        # bev_image = (bev_image - bev_image.min()) / (bev_image.max() - bev_image.min()) * 255
        # bev_image = bev_image.astype(np.uint8)

        # # PIL로 변환 및 저장
        # image = Image.fromarray(bev_image)
        # image.save(save_path_1)
