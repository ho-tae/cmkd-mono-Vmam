import os

import torch
import torch.nn as nn
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..backbones_3d.vfe.vf_encoder import MultiFusionVoxel
from ..backbones_3d.sst_v2 import SSTv2
from ..backbones_3d.sst_input_layer_v2 import SSTInputLayerV2
#from ..backbones_3d.multi_mae_sst_separate_top_only import MultiMAESSTSPChoose
from ..model_utils import model_nms_utils

class Detector3DTemplate_voxel_mae(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d', 'dense_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.model_cfg.TRAIN_CFG.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.model_cfg.TRAIN_CFG.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor,
            'train_cfg': self.model_cfg.TRAIN_CFG,
            'test_cfg': self.model_cfg.TEST_CFG
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            if module_name == "backbone_3d" and len(module) > 1:
                for m in module:
                    self.add_module(module_name, m)
            else:
                if module == None:
                    self.add_module(module_name, module)
                else:
                    self.add_module(module_name, module[0])
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        
        if self.model_cfg.VFE.NAME == "MultiSubVoxelDynamicVoxelNetSSL":
            vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
                loss_ratio_low=self.model_cfg.TRAIN_CFG.loss_ratio_low,
                loss_ratio_med=self.model_cfg.TRAIN_CFG.loss_ratio_med,
                loss_ratio_top=self.model_cfg.TRAIN_CFG.loss_ratio_top,
                loss_ratio_low_nor=self.model_cfg.TRAIN_CFG.loss_ratio_low_nor,
                loss_ratio_med_nor=self.model_cfg.TRAIN_CFG.loss_ratio_med_nor,
                loss_ratio_top_nor=self.model_cfg.TRAIN_CFG.loss_ratio_top_nor,
                sub_voxel_size_low=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_size_low),
                sub_voxel_size_med=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_size_med),
                sub_voxel_size_top=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_size_top),
                cls_loss_ratio_low=self.model_cfg.TRAIN_CFG.cls_loss_ratio_low,
                cls_loss_ratio_med=self.model_cfg.TRAIN_CFG.cls_loss_ratio_med,
                random_mask_ratio=self.model_cfg.TRAIN_CFG.random_mask_ratio,
                voxel_size=tuple(model_info_dict['voxel_size']),
                point_cloud_range=list(model_info_dict['point_cloud_range']),
                grid_size=model_info_dict['grid_size'],
                sub_voxel_ratio_low=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_ratio_low),
                sub_voxel_ratio_med=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_ratio_med),
            )

            model_info_dict['module_list'].append(vfe_module)
            return [vfe_module], model_info_dict
        
        elif self.model_cfg.VFE.NAME == "MultiFusionVoxel":
            vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
                sub_voxel_size_low=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_size_low),
                sub_voxel_size_med=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_size_med),
                voxel_size=tuple(self.model_cfg.TRAIN_CFG.voxel_size),
                point_cloud_range=list(model_info_dict['point_cloud_range']),
                grid_size=model_info_dict['grid_size'],
            )
            model_info_dict['module_list'].append(vfe_module)
            return [vfe_module], model_info_dict
        
        else: 
            vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
                num_point_features=model_info_dict['num_rawpoint_features'],
                point_cloud_range=list(model_info_dict['point_cloud_range']),
                voxel_size=tuple(model_info_dict['voxel_size']),
                grid_size=model_info_dict['grid_size'],
                depth_downsample_factor=model_info_dict['depth_downsample_factor'],
                return_gt_points=model_info_dict["train_cfg"]['return_gt_points']
            )
            model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
            model_info_dict['module_list'].append(vfe_module)
            return [vfe_module], model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        
        drop_info = list(self.model_cfg.BACKBONE_3D.drop_info_training.values()), list(self.model_cfg.BACKBONE_3D.drop_info_test.values())
        drop_info = (
            [{'max_tokens': x['max_tokens'], 'drop_range': tuple(x['drop_range'])} for x in drop_info[0]],
            [{'max_tokens': x['max_tokens'], 'drop_range': tuple(x['drop_range'])} for x in drop_info[1]]
        )
        
        drop_info_dict = {}
        for i, sublist in enumerate(drop_info):
            sub_dict = {}
            for j, item in enumerate(sublist):
                sub_dict[j] = item
            drop_info_dict[i] = sub_dict
        
        tuple_drop_info = (drop_info_dict[0], drop_info_dict[1])
        if self.model_cfg.BACKBONE_3D.NAME == 'MultiMAESSTSPChoose':
            backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            window_shape=tuple(self.model_cfg.BACKBONE_3D.WINDOW_SHAPE),
            shifts_list=[(0, 0), (self.model_cfg.BACKBONE_3D.WINDOW_SHAPE[0]//2, self.model_cfg.BACKBONE_3D.WINDOW_SHAPE[1]//2)],
            point_cloud_range=list(model_info_dict['point_cloud_range']),
            voxel_size=tuple(model_info_dict['voxel_size']),
            d_model=self.model_cfg.TRAIN_CFG.d_model * self.model_cfg.TRAIN_CFG.encoder_num,
            nhead=[8,]* self.model_cfg.TRAIN_CFG.encoder_num,
            dim_feedforward=self.model_cfg.TRAIN_CFG.dim_forward * self.model_cfg.TRAIN_CFG.encoder_num,
            sub_voxel_ratio_low=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_ratio_low),
            sub_voxel_ratio_med=tuple(self.model_cfg.TRAIN_CFG.sub_voxel_ratio_med),
            output_shape=list(self.model_cfg.TRAIN_CFG.grid_size[1:]),
            drop_info=tuple_drop_info,
        )
            model_info_dict['module_list'].append(backbone_3d_module)
            return [backbone_3d_module], model_info_dict
        
        if self.model_cfg.BACKBONE_3D.NAME == 'SSTInputLayerV2Masked':
            backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            sparse_shape=tuple(self.model_cfg.TRAIN_CFG.grid_size),
            voxel_size=tuple(model_info_dict['voxel_size']),
            drop_info=tuple_drop_info,
            window_shape=tuple(self.model_cfg.BACKBONE_3D.WINDOW_SHAPE)
        )
            
            sst_backbone_module = SSTv2(d_model=[128, ] * self.model_cfg.TRAIN_CFG.encoder_num,
                                        nhead=[8,]* self.model_cfg.TRAIN_CFG.encoder_num,
                                        num_blocks=self.model_cfg.TRAIN_CFG.encoder_num, 
                                        dim_feedforward=[256,] * self.model_cfg.TRAIN_CFG.encoder_num,
                                        output_shape=tuple(self.model_cfg.TRAIN_CFG.grid_size[:-1]), #[640, 640],
                                        num_attached_conv=0,
                                        conv_kwargs=[
                                            dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                            dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                            dict(kernel_size=3, dilation=2, padding=2, stride=1),
                                            ],
                                        conv_in_channel=128,
                                        conv_out_channel=128,
                                        debug=True,
                                        masked=True,
                                        )
            
            sst_decoder_module = SSTv2Decoder(d_model=[128, ]* self.model_cfg.TRAIN_CFG.encoder_num, 
                                              nhead=[8,]* self.model_cfg.TRAIN_CFG.encoder_num,
                                              num_blocks=self.model_cfg.TRAIN_CFG.encoder_num,  
                                              dim_feedforward=[256,]* self.model_cfg.TRAIN_CFG.encoder_num,
                                              output_shape=tuple(self.model_cfg.TRAIN_CFG.grid_size[:-1]),
                                              debug=True,
                                              use_fake_voxels=False,
                                              )
            model_info_dict['module_list'].append(backbone_3d_module)
            model_info_dict['module_list'].append(sst_backbone_module)
            model_info_dict['module_list'].append(sst_decoder_module)
            return [backbone_3d_module, sst_backbone_module, sst_decoder_module], model_info_dict
        
        if self.model_cfg.BACKBONE_3D.NAME == "SSTInputLayerV2":
            backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
                drop_info=tuple_drop_info,
                window_shape=tuple(self.model_cfg.BACKBONE_3D.WINDOW_SHAPE),
                sparse_shape=tuple(self.model_cfg.TRAIN_CFG.grid_size),
            )
            sst_backbone_module = SSTv2(d_model=[128, ] * self.model_cfg.TRAIN_CFG.encoder_num,
                            nhead=[8, ]* self.model_cfg.TRAIN_CFG.encoder_num,
                            encoder_num_blocks=self.model_cfg.TRAIN_CFG.encoder_num, 
                            dim_feedforward=[256,] * self.model_cfg.TRAIN_CFG.encoder_num,
                            output_shape=tuple(self.model_cfg.TRAIN_CFG.grid_size[:-1]),
                            num_attached_conv=3,
                            conv_kwargs=[
                                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                                dict(kernel_size=3, dilation=2, padding=2, stride=1),
                                ],
                            conv_in_channel=128,
                            conv_out_channel=128,
                            debug=True,
                            layer_cfg=dict(use_bn=False, cosine=True, tau_min=0.01),
                            checkpoint_blocks=[], # Consider removing it if the GPU memory is suffcient
                            conv_shortcut=True,
                            masked=False,
                            )
            
            model_info_dict['module_list'].append(backbone_3d_module)
            model_info_dict['module_list'].append(sst_backbone_module)
            model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
                if hasattr(backbone_3d_module, 'backbone_channels') else None
            return [backbone_3d_module, sst_backbone_module], model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return [map_to_bev_module], model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return [backbone_2d_module], model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        
        if self.model_cfg.BACKBONE_3D.NAME == 'SSTInputLayerV2Masked':
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=128,
                train_cfg=model_info_dict['train_cfg'],
                test_cfg=model_info_dict['test_cfg'],
                loss_weights=self.model_cfg['DENSE_HEAD']['LOSS_WEIGHTS'],
            )
        else:
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                class_names=self.class_names,
                grid_size=np.array(model_info_dict['grid_size']),
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
                voxel_size=model_info_dict.get('voxel_size', False)
            )
        model_info_dict['module_list'].append(dense_head_module)
        return [dense_head_module], model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                    
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
    
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        #for key in state_dict:
        #    if key not in update_model_state:
        #        logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch