import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from ..util.scatter_point import DynamicScatter
from pcdet.utils.spconv_utils import spconv as spconv
from ..util.utils import VFELayer, DynamicVFELayer, get_paddings_indicator
from pcdet.ops.sst.sst_ops import scatter_v2


class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 num_point_features=4, #in_channels=4,
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 voxel_size=(0.2, 0.2, 4),
                 grid_size=None,
                 depth_downsample_factor=None,
                 feat_channels=[64, 128],
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_gt_points=True
                 ):
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            num_point_features += 3
        if with_voxel_center:
            num_point_features += 3
        if with_distance:
            num_point_features += 3

        self.in_channels = num_point_features
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.depth_downsample_factor = depth_downsample_factor
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.return_gt_points = return_gt_points
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.voxel_layer = Voxelization(self.voxel_size, self.point_cloud_range, -1, (-1, -1))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
    
    def get_output_feature_dim(self):
        return self.in_channels
    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.
        3D 공간을 격자로 분할하여 캔버스를 생성
        
        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2] + 0.1) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point
    
    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                batch_dict,
                **kwargs
                ):
                #features,
                #coors,
                #points=None,
                #img_feats=None,
                #img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).[Nx4] [b,z,y,x]
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        
        points = batch_dict["points"]
        coors = []
        for batch_idx in range(batch_dict['batch_size']):
            mask = points[:, 0] == batch_idx
            batched_points = points[mask, 1:]
            res_coors = self.voxel_layer(batched_points)
            coors.append(res_coors)
        
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        points = points[:, 1:]
        
        features, coors = points, coors_batch # [154559, 4], [154559, 4]
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # features.requires_grad = True # for cam vis
        # features.register_hook(append_grad(-1)) # for cam vis
        # point_feat_dict[-1] = features.detach() # for cam vis

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            # point_feats.register_hook(append_grad(i)) # for cam vis
            # point_feat_dict[i] = point_feats.detach() # for cam vis

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        if self.return_gt_points:
            return voxel_feats, voxel_coors, low_level_point_feature, coors
        return voxel_feats, voxel_coors


class DynamicScatterVFE(DynamicVFE):
    """ Same with DynamicVFE but use torch_scatter to avoid construct canvas in map_voxel_center_to_point.
    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(self,
                 num_point_features=4, #in_channels=4,
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 voxel_size=(0.2, 0.2, 4),
                 grid_size=None,
                 depth_downsample_factor=None,
                 feat_channels=[64, 128],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_gt_points=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 unique_once=False,
                 ):
        super(DynamicScatterVFE, self).__init__(
            num_point_features,
            point_cloud_range,
            voxel_size,
            grid_size,
            depth_downsample_factor,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.unique_once = unique_once

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).[Nx4] [b,z,y,x]
            points (list[torch.Tensor], optional): Raw points used to guide the multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        
        if self.unique_once:
            new_coors, unq_inv_once = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv_once = None

        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, _, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', new_coors=new_coors,
                                                unq_inv=unq_inv_once)
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster / self.rel_dist_scaler)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                    coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                    coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                    coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, new_coors=new_coors,
                                                           unq_inv=unq_inv_once)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats

        if return_inv:
            return voxel_feats, voxel_coors, low_level_point_feature, unq_inv
        else:
            return voxel_feats, voxel_coors
    