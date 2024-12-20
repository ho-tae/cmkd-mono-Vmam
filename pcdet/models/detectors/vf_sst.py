from .detector3d_template_voxel_mae import Detector3DTemplate_voxel_mae

class VF_SST(Detector3DTemplate_voxel_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if cur_module == self.module_list[0]: # MultiFusionVoxel
                voxel_features, feature_coors = cur_module(batch_dict)
            if cur_module == self.module_list[1]: # SSTInputLayerV2
                x = cur_module(voxel_features, feature_coors, top_mid_low_voxel_info=True)
            if cur_module == self.module_list[2]: # SSTv2
                batch_dict['encoded_spconv_tensor'] = cur_module(x, model_info="SA_CA") #"SA_CA_SA_V3", "SST")
            if cur_module == self.module_list[3]: # HeightCompression
                x = cur_module(batch_dict)
            if cur_module == self.module_list[4]: # BaseBEVBackbone
                x = cur_module(batch_dict)
            if cur_module == self.module_list[5]: # Centerhead
                x = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
