import torch

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn


class DepthFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, depth_downsample_factor, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.downsample_factor = depth_downsample_factor
        self.module_topology = [
            'ffn',
        ]
        self.build_modules()

    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_ffn(self):
        """
        Builds frustum feature network
        Returns:
            ffn_module: nn.Module, Frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

            
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
            **kwargs:
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        batch_dict = self.ffn(batch_dict)
        return batch_dict

    def get_loss(self):
        """
        Gets DDN loss
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ffn.get_loss()
        return loss, tb_dict
