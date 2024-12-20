#from mmdet.models import BACKBONES
import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from pcdet.models.sst.sst_basic_block_v2 import BasicShiftBlockV2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SSTv2(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checkpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        encoder_num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        num_attached_conv=2,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        layer_cfg=dict(),
        conv_shortcut=False,
        masked=False,
        ):
        super().__init__()
        
        self.d_model = d_model
        #self.d_model_cat = [x * 3 for x in self.d_model]
        #self.dim_feedforward_cat = [x * 2 for x in dim_feedforward]
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.conv_shortcut = conv_shortcut

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0], bias=False)

        # Sparse Regional Attention Blocks(논문에서 말하는 SRA 모듈)
        encoder_block_list=[]
        encoder_cat_block_list=[]
        for i in range(encoder_num_blocks):
            encoder_block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )
            
        #for i in range(encoder_num_blocks):
        #    encoder_cat_block_list.append(
        #        BasicShiftBlockV2(self.d_model_cat[i], nhead[i], self.dim_feedforward_cat[i],
        #            dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
        #    )

        self.encoder_blocks = nn.ModuleList(encoder_block_list)
        #self.encoder_cat_block_list = nn.ModuleList(encoder_cat_block_list)   
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug

        self.masked = masked

        self.num_attached_conv = num_attached_conv

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                    )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)
            
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, voxel_info, model_info):
        '''
        '''
        num_shifts = 2
        if model_info=="SST":
            '''
            #assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

            device = voxel_info[0]['voxel_coors'].device
            batch_size = voxel_info[0]['voxel_coors'][:, 0].max().item() + 1
            voxel_feats_info = []
            for i in range(2):
                voxel_feat = voxel_info[i]['voxel_feats']
                ind_dict_list = [voxel_info[i][f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
                padding_mask_list = [voxel_info[i][f'key_mask_shift{i}'] for i in range(num_shifts)]
                pos_embed_list = [voxel_info[i][f'pos_dict_shift{i}'] for i in range(num_shifts)]
                voxel_feats_info.append([voxel_feat, ind_dict_list, padding_mask_list, pos_embed_list])
                
            voxel_feats_info = voxel_feats_info * 3
            output = voxel_feats_info[0][0]
            if hasattr(self, 'linear0'):
                output = self.linear0(output)
            for i, block in enumerate(self.encoder_blocks): # SRA(Sparse Regional Attention)
                output = block(output, voxel_feats_info[i][3], voxel_feats_info[i][1], 
                    voxel_feats_info[i][2], using_checkpoint = i in self.checkpoint_blocks)
            '''
            assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

            device = voxel_info['voxel_coors'].device
            batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
            voxel_feat = voxel_info['voxel_feats']
            ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
            pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            output = voxel_feat
            for i, block in enumerate(self.encoder_blocks):
                output = block(output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks) 
            
        if model_info=="SA_CA":
            # voxel_low_features = voxel_info["low_voxel_feats"]
            # voxel_mid_features = voxel_info["mid_voxel_feats"]
            
            low_mid_feat = voxel_info["mid_low_voxel_feats"]
            
            assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

            device = voxel_info['voxel_coors'].device
            batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
            voxel_feat = voxel_info['voxel_feats']
            ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
            pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            # top_output = voxel_feat
            # for i, block in enumerate(self.encoder_blocks):
            #     top_output = block(top_output, pos_embed_list, ind_dict_list, padding_mask_list)    
            
            # after_output_1 = voxel_low_features
            # for i, block in enumerate(self.encoder_blocks):
            #     if i != 0:
            #         after_output_1 = output_mb
            #         output_mb = torch.cat([before_output_1, after_output_1])
            #         before_output_1 = after_output_1
            #     else:
            #         before_output_1 = voxel_mid_features
            #         output_mb = torch.cat([before_output_1, after_output_1])
            #     output_mb = block(output_mb, pos_embed_list, ind_dict_list, padding_mask_list, cross_atten=True)
            
            # after_output_2 = output_mb
            
            # for i, block in enumerate(self.encoder_blocks):
            #     if i != 0:
            #         after_output_2 = output
            #         output = torch.cat([before_output_2, after_output_2])
            #         before_output_2 = after_output_2
            #     else:
            #         before_output_2 = top_output
            #         output = torch.cat([before_output_2, after_output_2])
            #     output = block(output, pos_embed_list, ind_dict_list, padding_mask_list, cross_atten=True)

            top_output = voxel_feat
            for i, block in enumerate(self.encoder_blocks):
                top_output = block(top_output, pos_embed_list, ind_dict_list, padding_mask_list)
                
            mid_low_output = low_mid_feat
            for i, block in enumerate(self.encoder_blocks):
                mid_low_output = block(mid_low_output, pos_embed_list, ind_dict_list, padding_mask_list)
                
            after_output_1 = mid_low_output
            
            for i, block in enumerate(self.encoder_blocks):
                if i != 0:
                    after_output_1 = output
                    output = torch.cat([before_output_1, after_output_1])
                    before_output_1 = after_output_1
                else:
                    before_output_1 = top_output
                    output = torch.cat([before_output_1, after_output_1])
                output = block(output, pos_embed_list, ind_dict_list, padding_mask_list, cross_atten=True)
                    
        if model_info=="SA_CA_SA":
            assert voxel_info[0]['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

            device = voxel_info[0]['voxel_coors'].device
            batch_size = voxel_info[0]['voxel_coors'][:, 0].max().item() + 1
            voxel_feat = voxel_info[0]['voxel_feats']
            ind_dict_list = [voxel_info[0][f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            padding_mask_list = [voxel_info[0][f'key_mask_shift{i}'] for i in range(num_shifts)]
            pos_embed_list = [voxel_info[0][f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            low_mid_feat = voxel_info[1]["mid_low_voxel_feats"]
            top_output = voxel_feat
            for i, block in enumerate(self.encoder_blocks):
                top_output = block(top_output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)    
                
            mid_low_output = low_mid_feat
            for i, block in enumerate(self.encoder_blocks):
                mid_low_output = block(mid_low_output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)   
            
            for i, block in enumerate(self.encoder_blocks[:3]):                
                if i != 0:
                    after_output = output_1
                    output_1 = torch.cat([before_output, after_output])
                    before_output = after_output
                else:
                    before_output = top_output
                    output_1 = torch.cat([before_output, mid_low_output])
                output_1 = block(output_1, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks, cross_atten=True)
            
            for i, block in enumerate(self.encoder_blocks[:3]):                
                if i != 0:
                    after_output_2 = output_2
                    output_2 = torch.cat([before_output_2, after_output_2])
                    before_output_2 = after_output
                else:
                    before_output_2 = mid_low_output
                    output_2 = torch.cat([before_output_2, top_output])
                output_2 = block(output_2, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks, cross_atten=True)
            
            ind_dict_list_2 = [voxel_info[1][f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            padding_mask_list_2 = [voxel_info[1][f'key_mask_shift{i}'] for i in range(num_shifts)]
            pos_embed_list_2 = [voxel_info[1][f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            output = torch.cat([output_1, output_2], dim=1)
            
            for i, block in enumerate(self.encoder_cat_block_list[:3]):
                output = block(output, pos_embed_list_2, ind_dict_list_2,
                    padding_mask_list_2, using_checkpoint = i in self.checkpoint_blocks)   
        
            if not self.masked:
                output = self.recover_bev(output, voxel_info[0]['voxel_coors'], batch_size)
                output = output.to(device)
                output_list = []
                if self.num_attached_conv > 0:
                    for conv in self.conv_layer.to(device):
                        temp = conv(output.to(device))
                        if temp.shape == output.shape:
                            output = temp + output
                        else:
                            output = temp
                        
                output_list.append(output)
                return output_list
            else:
                voxel_info[0]["output"] = output
                return voxel_info[0]
            
        if model_info=="SA_CA_SA_V3":
            assert voxel_info[0]['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

            device = voxel_info[0]['voxel_coors'].device
            batch_size = voxel_info[0]['voxel_coors'][:, 0].max().item() + 1
            voxel_feat = voxel_info[0]['voxel_feats']
            ind_dict_list = [voxel_info[0][f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            padding_mask_list = [voxel_info[0][f'key_mask_shift{i}'] for i in range(num_shifts)]
            pos_embed_list = [voxel_info[0][f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            top_output = voxel_feat
            for i, block in enumerate(self.encoder_blocks):
                top_output = block(top_output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)    
                
            mid_output = voxel_info[0]["mid_voxel_feats"]
            for i, block in enumerate(self.encoder_blocks):
                mid_output = block(mid_output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)   
                
            low_output = voxel_info[0]["low_voxel_feats"]
            for i, block in enumerate(self.encoder_blocks):
                low_output = block(low_output, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)
            
            for i, block in enumerate(self.encoder_blocks[:3]):                
                if i != 0:
                    after_output = output_1
                    output_1 = torch.cat([before_output, after_output])
                    before_output = after_output
                else:
                    before_output = top_output
                    output_1 = torch.cat([before_output, mid_output])
                output_1 = block(output_1, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks, cross_atten=True)
            
            for i, block in enumerate(self.encoder_blocks[:3]):                
                if i != 0:
                    after_output_2 = output_2
                    output_2 = torch.cat([before_output_2, after_output_2])
                    before_output_2 = after_output
                else:
                    before_output_2 = mid_output
                    output_2 = torch.cat([before_output_2, low_output])
                output_2 = block(output_2, pos_embed_list, ind_dict_list,
                    padding_mask_list, using_checkpoint = i in self.checkpoint_blocks, cross_atten=True)
            
            #for i, block in enumerate(self.encoder_blocks[:3]):                
            #    if i != 0:
            #        after_output_3 = output_3
            #        output_3 = torch.cat([before_output_3, after_output_3])
            #        before_output_3 = after_output
            #    else:
            #        before_output_3 = top_output
            #        output_3 = torch.cat([before_output_3, low_output])
            #    output_3 = block(output_3, pos_embed_list, ind_dict_list,
            #        padding_mask_list, using_checkpoint = i in self.checkpoint_blocks, cross_atten=True)
            
            output = torch.cat([output_1, output_2], dim=1)
                        
            #ind_dict_list_2 = [voxel_info[1][f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
            #padding_mask_list_2 = [voxel_info[1][f'key_mask_shift{i}'] for i in range(num_shifts)]
            #pos_embed_list_2 = [voxel_info[1][f'pos_dict_shift{i}'] for i in range(num_shifts)]
            
            #output = torch.cat([output_1, output_2], dim=1)
            
            #for i, block in enumerate(self.encoder_cat_block_list[:3]):
            #    output = block(output, pos_embed_list_2, ind_dict_list_2,
            #        padding_mask_list_2, using_checkpoint = i in self.checkpoint_blocks)   
        
            if not self.masked:
                output = self.recover_bev(output, voxel_info[0]['voxel_coors'], batch_size)
                output = output.to(device)
                output_list = []
                if self.num_attached_conv > 0:
                    for conv in self.conv_layer:
                        temp = conv(output.to(device))
                        if temp.shape == output.shape:
                            output = temp + output
                        else:
                            output = temp
                        
                output_list.append(output)
                return output_list
            else:
                voxel_info[0]["output"] = output
                return voxel_info[0]
        
        # If masked we want to send the output to the decoder and not a FPN how requires dense bev image
        if not self.masked:
            output = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
            output = output.to(device)
            output_list = []
            if self.num_attached_conv > 0:
                
                for conv in self.conv_layer:
                    temp = conv(output.to(device))
                #for conv in self.conv_layer.to(device):
                #    temp = conv(output.to(device))
                    if temp.shape == output.shape:
                        output = temp + output
                    else:
                        output = temp
                        
            output_list.append(output)
            return output_list
        else:
            voxel_info["output"] = output
            return voxel_info
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas