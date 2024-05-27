# import os
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.pascal_voc import register_pascal_voc
# os.environ['DETECTRON2_DATASETS'] = '/home/add_disk_e/dataset/VOCdevkit/'

# def register_all_voc(root):
#     SPLITS = [
#         ("voc_2007_trainval", "VOC2007", "trainval"),
#         ("voc_2007_test", "VOC2007", "test"),
#         ("voc_2012_trainval", "VOC2012", "trainval"),
#     ]
    
#     for name, dirname, split in SPLITS:
#         # if name not in DatasetCatalog.list():
#         year = 2007 if "2007" in name else 2012
#         register_pascal_voc(name, os.path.join(root, dirname), split, year)
#         MetadataCatalog.get(name).evaluator_type = "pascal_voc"

# # 使用环境变量获取数据集根目录
# root = os.environ['DETECTRON2_DATASETS']
# register_all_voc(root)

import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import xml.etree.ElementTree as ET

def load_voc_instances(dirname: str, split: str):
    fileids = []
    with open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = [line.strip() for line in f]

    dataset_dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)
        root = tree.getroot()

        record = {}
        record["file_name"] = jpeg_file
        record["image_id"] = fileid
        record["height"] = int(root.find("size").find("height").text)
        record["width"] = int(root.find("size").find("width").text)

        objs = []
        for obj in root.findall("object"):
            obj_struct = {}
            obj_struct["bbox"] = [int(obj.find("bndbox").find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            obj_struct["bbox_mode"] = BoxMode.XYXY_ABS
            obj_struct["category_id"] = obj.find("name").text
            objs.append(obj_struct)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

from detectron2.data import MetadataCatalog, DatasetCatalog

def register_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=["aeroplane", "bicycle", "bird", "boat", "bottle",
                       "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant",
                       "sheep", "sofa", "train", "tvmonitor"],
        dirname=dirname,
        split=split,
        year=year,
    )

register_voc("voc_2012_test", "/home/add_disk_e/dataset/VOCdevkit/VOC2012", "test", 2012)

from detectron2.data import DatasetCatalog, MetadataCatalog

# 查看数据集的样本
dataset_dicts = DatasetCatalog.get("voc_2012_test")
print("Dataset sample:", dataset_dicts[0])

# 查看元数据
voc_metadata = MetadataCatalog.get("voc_2012_test")
print("Metadata:", voc_metadata)
