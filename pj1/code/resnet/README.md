
# 使用示例，根据实际路径修改

## envrionment
cd /home/fudan248/zhangjinyu/code_repo/DATA620004-SS24/pj1/code/resnet
conda activate mmdetection

## train
python code/train.py

## test
python code/test.py --checkpoint_path ckpts/model_resnet18+pretrained_True+fc_lr_0.01+pretrained_lr_0.0001+momentum_0.9+augment_False+weight_decay_0.001+dropout_0+scheduler_StepLR+step_size_10+num_epochs_300.pth