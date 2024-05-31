import os
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_pascal_voc
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# 配置模型
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection/configs/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("voc_2012_trainval",)
    cfg.DATASETS.TEST = ("voc_2012_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"  # 初始化预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 8  # 每个batch包含16张图片
    cfg.SOLVER.BASE_LR = 0.02  # 根据GPU数量调整学习率
    cfg.SOLVER.MAX_ITER = 180000  # 根据数据集大小和GPU数量调整迭代次数
    cfg.SOLVER.STEPS = (120000, 160000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # VOC 数据集有20个类别
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

cfg = setup_cfg()

# 训练模型
def main():
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# 使用 launch 函数指定使用 8 张 GPU 训练
if __name__ == "__main__":
    launch(
        main,
        num_gpus_per_machine=8,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:12345",  # 修改为适当的地址和端口
    )

# # 测试模型
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试时的阈值
# predictor = DefaultPredictor(cfg)

# 评估模型
# evaluator = COCOEvaluator("voc_2012_test", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "voc_2012_test")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))

