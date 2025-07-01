import cv2
import numpy as np
import os
from tqdm import tqdm

def load_kitti_txt_result(txt_path):
    detections = []
    if not os.path.exists(txt_path):
        return detections  # 예측 결과 없는 경우 무시
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            if len(data) < 15:
                continue
            detection = {
                'type': data[0],
                'bbox': list(map(float, data[4:8])),  # [left, top, right, bottom]
                'score': float(data[-1])
            }
            detections.append(detection)
    return detections

def visualize_and_save(image_path, txt_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    detections = load_kitti_txt_result(txt_path)
    for det in detections:
        bbox = det['bbox']
        label = f"{det['type']} {det['score']:.2f}"
        color = (0, 255, 0)
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, label, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

# 전체 이미지 반복 처리
image_dir = '/cmkd-mono-Vmam/data/kitti/testing/image_2/'
txt_dir = '/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_entire_train_occupancy/default/eval/epoch_80/test/default/final_result/data/'
output_dir = '/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_entire_train_occupancy/default/eval/epoch_80/test/default/final_result/data/visual/'

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

for image_file in tqdm(image_files):
    image_id = image_file.replace('.png', '')
    image_path = os.path.join(image_dir, image_file)
    txt_path = os.path.join(txt_dir, f'{image_id}.txt')
    output_path = os.path.join(output_dir, f'{image_id}_vis.png')
    visualize_and_save(image_path, txt_path, output_path)