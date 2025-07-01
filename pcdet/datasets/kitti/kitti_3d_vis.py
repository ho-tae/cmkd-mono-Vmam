import os
import numpy as np
import cv2
from tqdm import tqdm

def read_calib_file(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    calib = {}
    for line in lines[:-1]:
        key, value = line.strip().split(':', 1)
        calib[key] = np.array([float(x) for x in value.strip().split()])
    P2 = calib['P2'].reshape(3, 4)
    return P2

def compute_box_3d(dim, location, ry):
    h, w, l = dim
    x, y, z = location

    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ np.vstack([x_corners, y_corners, z_corners])
    corners_3d += np.array([[x], [y], [z]])
    return corners_3d.T

def project_to_image(pts_3d, P):
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d = pts_3d_homo @ P.T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2]

def draw_projected_box3d(image, qs, color=(0,255,0), thickness=2):
    qs = qs.astype(int)
    # 4 lines around the bottom face
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, tuple(qs[i]), tuple(qs[j]), color, thickness)
    # 4 lines around the top face
    for k in range(4, 8):
        i, j = k, 4 + (k + 1) % 4
        cv2.line(image, tuple(qs[i]), tuple(qs[j]), color, thickness)
    # 4 vertical lines
    for k in range(4):
        cv2.line(image, tuple(qs[k]), tuple(qs[k+4]), color, thickness)
    return image

def visualize_3d_bbox(image_path, txt_path, calib_path, save_path):
    image = cv2.imread(image_path)
    if not os.path.exists(txt_path):
        return
    P2 = read_calib_file(calib_path)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 15:
            continue
        dim = [float(data[8]), float(data[9]), float(data[10])]  # h, w, l
        loc = [float(data[11]), float(data[12]), float(data[13])]
        ry = float(data[14])
        corners_3d = compute_box_3d(dim, loc, ry)
        box_2d = project_to_image(corners_3d, P2)
        image = draw_projected_box3d(image, box_2d)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)

# 예시 실행
image_dir = '/cmkd-mono-Vmam/data/kitti/training/image_2'
calib_dir = '/cmkd-mono-Vmam/data/kitti/training/calib'
txt_dir = '/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc/default/eval/epoch_80/val/default/final_result/data'#'/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc/default/eval/epoch_80/test/default/final_result/data'
save_dir = '/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc/default/eval/epoch_80/val/default/final_result/3d_vis'#'/cmkd-mono-Vmam/output/cmkd-mono-Vmam/tools/cfgs/kitti_models/caddn_vmamba_occupancy_voxel_sc/default/eval/epoch_80/test/default/final_result/3d_vis'

for fname in tqdm(sorted(os.listdir(image_dir))):
    if not fname.endswith('.png'): continue
    image_id = fname.replace('.png', '')
    image_path = os.path.join(image_dir, fname)
    txt_path = os.path.join(txt_dir, f'{image_id}.txt')
    calib_path = os.path.join(calib_dir, f'{image_id}.txt')
    save_path = os.path.join(save_dir, f'{image_id}_3d.png')
    visualize_3d_bbox(image_path, txt_path, calib_path, save_path)
