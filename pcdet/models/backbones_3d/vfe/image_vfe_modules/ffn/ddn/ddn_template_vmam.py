from collections import OrderedDict
from pathlib import Path
from torch import hub
from mmseg.apis import init_model, inference_model
#from .binsformer_head import BinsFormerDecodeHead
from .depth_head import DepthHead

import sys
import os

# Add the directory containing VMamba to the system path
sys.path.append('/cmkd-mono-Vmam/VMamba')

# Now you can import model from VMamba/segmentation
from segmentation import model
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNTemplateVMam(nn.Module):

    def __init__(self, backbone_name, image_feat_extract_layer, depth_feat_extract_layer, num_classes, config_path=None, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            backbone_name: string, Backbone model name
            image_feat_extract_layer: int, Layer to extract image features from
            depth_feat_extract_layer: int, Layer to extract depth features from
            num_classes: int, Number of classes
            config_path: string, (Optional) Path to configuration file
            pretrained_path: string, (Optional) Path to pretrained weights
            aux_loss: bool, Flag to include auxiliary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.config_path = config_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(backbone_name=backbone_name)
        self.image_feat_extract_layer = image_feat_extract_layer
        self.depth_feat_extract_layer = depth_feat_extract_layer
        
        self.depth_head = DepthHead(in_channels=128, out_channels=151)
        
        # self.binsformer = BinsFormerDecodeHead(
        #     transformer_encoder=dict( # default settings
        #                              type='PureMSDEnTransformer',
        #                              num_feature_levels=3,
        #                              encoder=dict(
        #                                  type='DetrTransformerEncoder',
        #                                  num_layers=6,
        #                                  transformerlayers=dict(
        #                                      type='BaseTransformerLayer',
        #                                      attn_cfgs=dict(
        #                                          type='MultiScaleDeformableAttention', 
        #                                          embed_dims=256, 
        #                                          num_levels=3, 
        #                                          num_points=8),
        #                                      feedforward_channels=1024,
        #                                      ffn_dropout=0.1,
        #                                      operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        #     positional_encoding=dict(
        #         type='SinePositionalEncoding', num_feats=128, normalize=True),
        #     transformer_decoder=dict(
        #         type='PixelTransformerDecoder',
        #         return_intermediate=True,
        #         num_layers=9,
        #         num_feature_levels=3,
        #         hidden_dim=256,
        #         operation='//',
        #         transformerlayers=dict(
        #             type='PixelTransformerDecoderLayer',
        #             attn_cfgs=dict(
        #                 type='MultiheadAttention',
        #                 embed_dims=256,
        #                 num_heads=8,
        #                 dropout=0.0),
        #             ffn_cfgs=dict(
        #                 feedforward_channels=2048,
        #                 ffn_drop=0.0),
        #             operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))),
        #     in_channels=[128, 256, 512, 1024],
        #     channels=256,
        #     n_bins=150,
        #     loss_decode={'type': 'SigLoss', 'valid_mask': True, 'loss_weight': 10},
        #     min_depth=2.0, # caddn 2.0 binsformer 0.001
        #     max_depth=46.8,)   # cadde 46.8 binsformer 10

        
    def get_model(self, backbone_name):
        """
        Get model
        Args:
            backbone_name: backbone_name
        Returns:
            model: nn.Module, Model
        """
        model = init_model(self.config_path, self.pretrained_path)            
        
        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            
            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = pretrained_dict["state_dict"]
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                                                          pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                            if "aux_classifier" not in key}

        # Removes conv_seg weights if number of classes are different
        if "decode_head.conv_seg.weight" in pretrained_dict:
            model_num_classes = model_dict["decode_head.conv_seg.weight"].shape[0]
            pretrained_num_classes = pretrained_dict["decode_head.conv_seg.weight"].shape[0]
            if model_num_classes != pretrained_num_classes:
                pretrained_dict.pop("decode_head.conv_seg.weight")
                pretrained_dict.pop("decode_head.conv_seg.bias")

        # Handles auxiliary head (FCNHead) if present
        if "auxiliary_head.conv_seg.weight" in pretrained_dict:
            model_aux_classes = model_dict["auxiliary_head.conv_seg.weight"].shape[0]
            pretrained_aux_classes = pretrained_dict["auxiliary_head.conv_seg.weight"].shape[0]
            if model_aux_classes != pretrained_aux_classes:
                pretrained_dict.pop("auxiliary_head.conv_seg.weight")
                pretrained_dict.pop("auxiliary_head.conv_seg.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns:
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features from image_feat_extract_layer
                logits: (N, num_classes, H_out, W_out), Classification logits
        """
        # Extract features
        result = OrderedDict()
        x = images
        
        features = self.model.backbone(x)                
        result['features'] = features[0]
        
        features = self.model(x)
        result["logits"] = features
        
        #result["logits"] = self.depth_head(result["features"])  # (N, 1, H, W)
        
        # pred_depths, pred_logit, pred_classes = self.binsformer(features)
        # result["pred_depths"] = pred_depths[-1]
        # result["pred_logit"] = pred_logit #[:, :-1, :, :]
        # result["pred_classes"] = pred_classes
    
        return result

    def preprocess(self, images): 
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        
        if self.pretrained:
            # Create a mask for padded pixels
            mask = (x == 0)

            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
