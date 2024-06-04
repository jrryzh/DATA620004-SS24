
# 项目说明

## 项目目录
- code: 存放所有代码
- sample_images: 存放detection任务的样本图片

## 环境配置
本项目依赖以下Python包，请确保在运行之前安装它们。

主要依赖
torch: 1.8.1+cu101
torchvision: 0.9.1+cu101
numpy: 1.24.4
pandas: 2.0.3
matplotlib: 3.7.5
scipy: 1.10.1
辅助库
absl-py: 2.1.0
addict: 2.4.0
antlr4-python3-runtime: 4.9.3
asttokens: 2.4.1
cffi: 1.16.0
cryptography: 42.0.7
decorator: 5.1.1
google-auth: 2.29.0
grpcio: 1.64.0
hydra-core: 1.3.2
opencv-python: 4.9.0.80
protobuf: 5.26.1
pycocotools: 2.0.7
requests: 2.28.2
tensorboard: 2.14.0
版本控制
git: 请确保安装了Git
安装依赖
可以通过以下命令安装所有依赖：

pip install -r requirements.txt
请确保您的Python版本与上述依赖兼容。

特别说明
detectron2 和 mmdetection 位于本地路径：
detectron2: /home/fudan248/zhangjinyu/code_repo/detectron2
mmdetection: /home/fudan248/zhangjinyu/code_repo/mmdetection
请根据实际情况调整本地路径或使用pip安装。


## 权重下载
resnet和fastercnn的权重：
https://drive.google.com/drive/folders/1NgfFNxPWOCpehrZDTMF-Gk1j-gbkutte?usp=drive_link
yolov3权重：
https://drive.google.com/drive/folders/1RF5v1ahVBJnV6Qh3TJVHsGoxiKG-P1vo?usp=sharing 