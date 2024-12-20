import torch
import torch.nn as nn
import numpy as np

from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from .dynamic_voxel_vfe_dymvr import DynamicVoxelVFE
from ..util.voxelize import VoxelizationByGridShape
from pcdet.ops.sst.sst_ops import scatter_v2

eps = 1e-9


class MultiFusionVoxel(nn.Module):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_."""

    def __init__(
        self,
        grid_size,
        voxel_size,
        point_cloud_range,
        sub_voxel_size_low,
        sub_voxel_size_med,
    ):
        super(MultiFusionVoxel, self).__init__()
        
        self.point_cloud_range = point_cloud_range
        
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.sub_voxel_size_low = sub_voxel_size_low
        self.sub_voxel_size_med = sub_voxel_size_med
        
        voxel_layer = dict(
            voxel_size=self.voxel_size,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
        )
        sub_voxel_layer_low = dict(
            voxel_size=self.sub_voxel_size_low,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
            )
        sub_voxel_layer_med = dict(
            voxel_size=self.sub_voxel_size_med,
            max_num_points=-1,
            point_cloud_range=self.point_cloud_range,
            max_voxels=(-1, -1),
            )
        
        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.sub_voxel_layer_low = VoxelizationByGridShape(**sub_voxel_layer_low)
        self.sub_voxel_layer_med = VoxelizationByGridShape(**sub_voxel_layer_med)
        
        self.voxel_encoder = DynamicVoxelVFE(
            num_point_features=4,
            voxel_size=self.voxel_size,
            grid_size = self.voxel_layer.grid_shape,
            point_cloud_range=self.point_cloud_range,
            num_filters=[64, 128],
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
        )
        
        self.voxel_encoder_middle = DynamicVoxelVFE(
            num_point_features=4,
            voxel_size=self.sub_voxel_size_med,
            point_cloud_range=self.point_cloud_range,
            grid_size = self.sub_voxel_layer_med.grid_shape,
            num_filters=[64, 128],
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
        )
        
        self.voxel_encoder_middle = DynamicVoxelVFE(
            num_point_features=4,
            voxel_size=self.sub_voxel_size_low,
            point_cloud_range=self.point_cloud_range,
            grid_size = self.sub_voxel_layer_low.grid_shape,
            num_filters=[64, 128],
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
        )
            
    def forward(self, batch_dict):
        """Extract features from points."""
        
        points = batch_dict["points"]
        batch_size = batch_dict["batch_size"]
        res_coors = []

        for batch_idx in range(batch_size):
            mask = points[:, 0] == batch_idx
            batched_points = points[mask, 1:]
            res_coors.append(batched_points)
    
        voxels, coors = self.voxelize(res_coors)
        sub_voxels_low, sub_coors_low = self.sub_voxelize_low(res_coors)
        sub_voxels_med, sub_coors_med = self.sub_voxelize_med(res_coors)
        
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        voxel_mid_features, feature_mid_coors = self.voxel_encoder_middle(sub_voxels_med, sub_coors_med)
        voxel_low_features, feature_low_coors = self.voxel_encoder_low(sub_voxels_low, sub_coors_low)

        # 중단 + 하단 voxel 결합
        mid_bottom_voxels = torch.cat([feature_mid_coors, feature_low_coors])[:, 1:]
        
        # 상단 voxel을 기준으로 중단 + 하단 voxel과의 관계 파악
        k = 24 # 상단 voxel에서 가장 가까운 16개의 중단/하단 voxel 탐색
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn.fit(mid_bottom_voxels.cpu())
        
        # 상단 voxel에서 가까운 중단/하단 voxel 찾기
        distances, indices = knn.kneighbors(feature_coors[:, 1:].cpu())
        
        # k-NN 특성 추출
        knn_features = torch.cat([voxel_mid_features, voxel_low_features], dim=0)[indices]
        
        # 6. 거리 기반 가중치 계산 (1 / (1 + 거리)) 및 가중 평균
        weights = 1 / (1 + torch.tensor(distances, dtype=torch.float32).to(knn_features.device))  # 거리 기반 가중치 계산
        weights = weights / weights.sum(dim=1, keepdims=True)  # 가중치 정규화 (각 샘플에 대해 합이 1)

        # 7. 가중 평균을 통한 특성 집합화
        weighted_knn_features = knn_features * weights.unsqueeze(-1)  # 가중치 적용
        aggregated_features = weighted_knn_features.sum(dim=1)  # [n, feature_dim] 형태로 집합화
        
        return [voxel_features, aggregated_features], feature_coors #[merge_voxel_mid_low_features], feature_coors # [merge_voxel_mid_low_features], feature_coors #[voxel_features, voxel_med_mean, voxel_low_mean], feature_coors
    
    @torch.no_grad()
    def sub_voxelize_low(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, res in enumerate(points):
            res_coors = self.sub_voxel_layer_low(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    def sub_voxelize_med(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, res in enumerate(points):
            res_coors = self.sub_voxel_layer_med(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch