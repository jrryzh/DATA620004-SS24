import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.data.transforms import ResizeShortestEdge
import cv2
import matplotlib.pyplot as plt
import random

# 配置
cfg = get_cfg()
cfg.merge_from_file("/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection2/configs/faster_rcnn_R_101_FPN_3x_gpt.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试阈值
cfg.MODEL.WEIGHTS = "/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection2/output_resnet101_fpn_3x_gpt/model_0019999.pth"  # 加载训练好的模型权重
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000  # 增加RPN生成的proposals数量

# 创建预测器
predictor = DefaultPredictor(cfg)

# 加载VOC2007测试集
dataset_dicts = DatasetCatalog.get("voc_2007_test")
MetadataCatalog.get("voc_2007_test").thing_classes = MetadataCatalog.get("voc_2007_test").thing_classes

# 随机挑选4张图像
random.seed(42)
selected_images = random.sample(dataset_dicts, 4)

# 进行预测和可视化
for d in selected_images:
    img = cv2.imread(d["file_name"])
    
    # 图像预处理
    transform_gen = ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    transformed_image = transform_gen.get_transform(img).apply_image(img)
    transformed_image = torch.as_tensor(transformed_image.transpose(2, 0, 1)).cuda()
    inputs = {"image": transformed_image, "height": img.shape[0], "width": img.shape[1]}
    
    with torch.no_grad():
        images = predictor.model.preprocess_image([inputs])
        features = predictor.model.backbone(images.tensor)
        proposals, _ = predictor.model.proposal_generator(images, features)
    
    # 获取最终的预测结果
    outputs = predictor(img)
    
    # 可视化RPN生成的proposals
    v_proposals = Visualizer(img[:, :, ::-1], MetadataCatalog.get("voc_2007_test"), scale=1.2)
    v_proposals = v_proposals.overlay_instances(boxes=proposals[0].proposal_boxes.tensor.cpu().numpy())
    
    # 可视化最终的预测结果
    v_predictions = Visualizer(img[:, :, ::-1], MetadataCatalog.get("voc_2007_test"), scale=1.2)
    v_predictions = v_predictions.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 使用matplotlib展示
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(v_proposals.get_image()[:, :, ::-1])
    ax[0].set_title('RPN Proposals')
    ax[0].axis('off')

    ax[1].imshow(v_predictions.get_image()[:, :, ::-1])
    ax[1].set_title('Final Predictions')
    ax[1].axis('off')
    
    plt.savefig(f"visual_output/{d['image_id']}.png")
    plt.close()
