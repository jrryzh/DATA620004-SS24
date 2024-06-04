# 使用示例，根据实际路径修改

## env
cd /home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/detection2
conda activate mmdetection

## train
### single-gpu
CUDA_VISIBLE_DEVICES=7 python code/plain_train_net.py --config-file configs/faster_rcnn_R_50_FPN_3x.yaml
CUDA_VISIBLE_DEVICES=6 python code/plain_train_net.py --config-file configs/faster_rcnn_R_101_FPN_3x.yaml
### multi-gpu
python code/plain_train_net.py --config-file configs/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 4

## eval
### single-gpu
python code/plain_train_net.py --config-file configs/faster_rcnn_R_101_FPN_3x.yaml --eval-only MODEL.WEIGHTS output_2007/model_0089999.pth
### multi-gpu
python code/plain_train_net.py --config-file configs/faster_rcnn_R_101_FPN_3x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS output_2007/model_0089999.pth