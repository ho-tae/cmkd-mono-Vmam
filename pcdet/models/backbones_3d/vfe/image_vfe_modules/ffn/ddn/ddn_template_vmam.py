from collections import OrderedDict
from pathlib import Path
from torch import hub
from mmseg.apis import init_model, inference_model

import sys
import os

# Add the directory containing VMamba to the system path
sys.path.append('/cmkd/VMamba')

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
                image_features: (N, C, H_out, W_out), Image features from image_feat_extract_layer
                depth_features: (N, C, H_out, W_out), Depth features from depth_feat_extract_layer
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxiliary classification logits
        """
        # Extract features
        result = OrderedDict()
        x = images
        
        features = self.model.backbone(x)
        result['features'] = features[0]
        feat_shape = result['features'].shape[-2:]
        
        #features = self.model.decode_head(features)
        
        # Prediction classification logits
        #x = features[3]
        
        features = self.model(x)
        # features = self.model.auxiliary_head(features)
        # features = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result["logits"] = features

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
