import pickle
import torch
import numpy as np
from pathlib import Path
from pcdet.datasets import build_dataloader
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common as kitti_utils

def create_kitti_infos(config_path, pkl_path, save_dir):
    # config 불러오기
    cfg_from_yaml_file(config_path, cfg)
    logger = common_utils.create_logger()
    
    # dataset 생성
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        logger=logger
    )
    
    # 예측 결과 불러오기
    with open(pkl_path, 'rb') as f:
        pred_dicts = pickle.load(f)

    # 배치용 dummy batch_dict 만들기 (frame_id, calib 등 필요)
    batch_dicts = []
    for idx in range(len(dataset)):
        info = dataset[idx]
        frame_id = info['frame_id']
        calib = info['calib']
        image_shape = info['image_shape']
        batch_dicts.append({
            'frame_id': frame_id,
            'calib': calib,
            'image_shape': torch.tensor(image_shape),
        })

    # KITTI 포맷 저장 경로
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 저장
    dataset.generate_prediction_dicts(
        batch_dict=batch_dicts,
        pred_dicts=pred_dicts,
        class_names=cfg.CLASS_NAMES,
        output_path=output_dir
    )
    print(f"[INFO] KITTI submission files saved to: {output_dir}")

if __name__ == '__main__':
    config_path = '/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc.yaml'  # 모델 설정 경로
    pkl_path = '/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc/default/eval/epoch_80/test/default/result.pkl'  # 결과 파일 경로
    save_dir = './kitti_submission'  # 저장할 폴더

    create_kitti_infos(config_path, pkl_path, save_dir)
