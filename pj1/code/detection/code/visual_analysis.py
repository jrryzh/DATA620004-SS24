import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.data import detection_utils as utils
import cv2
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_pascal_voc
from detectron2.data import DatasetCatalog, MetadataCatalog
import random

# 设置配置文件路径和模型权重路径
config_file_path = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/configs/faster_rcnn_R_101_FPN_3x.yaml"
model_weights_path = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/output_2007/model_0089999.pth"

# 加载配置文件
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试阈值
cfg.MODEL.DEVICE = "cuda"  # 使用 GPU 进行推理

# 创建预测器
predictor = DefaultPredictor(cfg)

def get_proposals_and_predictions(cfg, model, image):
    # 将图像转换为模型输入格式
    height, width = image.shape[:2]
    image_tensor = predictor.aug.get_transform(image).apply_image(image)
    image_tensor = torch.as_tensor(image_tensor.astype("float32").transpose(2, 0, 1))
    
    inputs = {"image": image_tensor, "height": height, "width": width}

    # 前向传播获取 RPN proposals 和最终预测结果
    with torch.no_grad():
        images = model.preprocess_image([inputs])
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        results, _ = model.roi_heads(images, features, proposals)
    
    return proposals[0], results[0]

# def visualize_proposals_and_predictions(image, proposals, predictions, metadata):
#     v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
#     v = v.overlay_instances(boxes=proposals.proposal_boxes)
#     proposals_img = v.get_image()[:, :, ::-1]

#     v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
#     v = v.draw_instance_predictions(predictions)
#     predictions_img = v.get_image()[:, :, ::-1]

#     return proposals_img, predictions_img

def visualize_proposals_and_predictions(image, proposals, predictions, metadata):
    proposals = proposals.to('cpu')
    predictions = predictions.to('cpu')
    
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.overlay_instances(boxes=proposals.proposal_boxes)
    proposals_img = v.get_image()[:, :, ::-1]

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.draw_instance_predictions(predictions)
    predictions_img = v.get_image()[:, :, ::-1]

    return proposals_img, predictions_img

def show_images(image_list, index, titles=None):
    plt.figure(figsize=(20, 10))
    for i, img in enumerate(image_list):
        plt.subplot(1, len(image_list), i + 1)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.savefig(f"/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/visual_output/output_{index}.png")

# 加载测试集
dataset_dicts = DatasetCatalog.get("voc_2007_test")
metadata = MetadataCatalog.get("voc_2007_test")

# 随机挑选4张图像
random.seed(42)
sampled_images = random.sample(dataset_dicts, 20)

# 加载模型
model = build_model(cfg)
DetectionCheckpointer(model).load(model_weights_path)
model.eval()

for index, d in enumerate(sampled_images):
    image = cv2.imread(d["file_name"])
    proposals, predictions = get_proposals_and_predictions(cfg, model, image)

    proposals_img, predictions_img = visualize_proposals_and_predictions(
        image, proposals, predictions, metadata
    )

    # 显示原始图像、proposals 和最终预测结果
    show_images([image[:, :, ::-1], proposals_img, predictions_img], index=index,
                titles=["Original Image", "RPN Proposals", "Final Predictions"])
