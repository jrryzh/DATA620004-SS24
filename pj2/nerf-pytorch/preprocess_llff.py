import numpy as np
import os
import imageio

def load_images(image_dir):
    # 读取图像文件，假设图像文件命名为 `image_*.jpg`
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    images = [imageio.imread(f) for f in image_files]
    return np.array(images)

def load_camera_params(camera_file):
    # 读取相机参数文件
    with open(camera_file, 'r') as f:
        lines = f.readlines()
    camera_params = []
    for line in lines:
        if line.startswith('#') or line.startswith('Number of cameras'):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        fx = float(parts[4])
        fy = float(parts[5])
        cx = float(parts[6])
        cy = float(parts[7])
        camera_params.append((camera_id, model, width, height, fx, fy, cx, cy))
    return camera_params

def load_poses(poses_file):
    # 读取相机位姿文件
    poses = np.loadtxt(poses_file)
    poses = poses.reshape(-1, 3, 4)
    return poses

def compute_bounds(poses, near=0.1, far=1.0):
    # 计算每个视角下的边界，假设near和far平面距离
    bounds = []
    for pose in poses:
        # 计算每个视锥体的边界
        bound = np.array([near, far])
        bounds.append(bound)
    return np.array(bounds)

def main(image_dir, camera_file, poses_file, output_file):
    # 读取图像数据
    images = load_images(image_dir)
    # 读取相机参数
    camera_params = load_camera_params(camera_file)
    # 打印相机参数供调试
    print(f"Camera Parameters: {camera_params}")
    # 读取相机位姿
    poses = load_poses(poses_file)
    # 计算边界
    bounds = compute_bounds(poses)
    # 合并位姿和边界
    poses_bounds = np.hstack([poses.reshape(-1, 12), bounds])
    # 保存为 .npy 文件
    np.save(output_file, poses_bounds)

if __name__ == "__main__":
    image_dir = './data/nerf_llff_data/bird/images'
    camera_file = './data/nerf_llff_data/bird/cameras.txt'
    poses_file = './data/nerf_llff_data/bird/poses.txt'
    output_file = './data/nerf_llff_data/bird/poses_bounds.npy'
    main(image_dir, camera_file, poses_file, output_file)
