import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt

# 配置模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/configs/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/output_2007/model_0089999.pth"  # 替换为你自己的模型路径
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # 设置测试的阈值
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 替换为你自己的类别数

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
    plt.show()

# 可视化三张图像的预测结果
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]

for image_path in image_paths:
    visualize_predictions(image_path)
