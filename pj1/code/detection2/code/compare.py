import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt
import os

# 配置模型
cfg = get_cfg()

# 方法一：使用Detectron2提供的默认配置文件
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

# 方法二：使用自定义配置文件
cfg.merge_from_file("/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection2/configs/faster_rcnn_R_101_FPN_3x_gpt.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试阈值
cfg.MODEL.WEIGHTS = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection2/output_resnet101_fpn_3x_gpt/model_0019999.pth"  # 加载训练好的模型权重

# 创建预测器
predictor = DefaultPredictor(cfg)

def visualize_predictions(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 进行预测
    outputs = predictor(img)
    
    # 可视化结果
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # 显示结果
    plt.figure(figsize=(14, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis("off")
    # plt.show()
    plt.savefig(os.path.join("visual_output", image_path.split("/")[-1].split(".")[0] + "_prediction.png"))

# 可视化三张图像的预测结果
image_root = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/sample_images"
images = os.listdir(image_root)
image_paths = [os.path.join(image_root, image) for image in images]

for image_path in image_paths:
    visualize_predictions(image_path)
