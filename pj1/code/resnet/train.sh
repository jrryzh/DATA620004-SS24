# test1 
# python train_from_pretrained_dataparallel.py \
#     --model resnet18 \
#     --pretrained True \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-3 \
#     --momentum 0.9 \
#     --num_epochs 50

# python train_from_pretrained_dataparallel.py \
#     --model resnet18 \
#     --pretrained True \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-4 \
#     --momentum 0.9 \
#     --num_epochs 50

# test2 
# python train_from_pretrained_dataparallel.py \
#     --model resnet50 \
#     --pretrained True \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-3 \
#     --momentum 0.9 \
#     --num_epochs 100

# add aug
python train_from_pretrained_dataparallel.py \
    --model resnet18 \
    --pretrained True \
    --data_dir /share/home/zjy/data/CUB_200_2011 \
    --batch_size 512 \
    --fc_learning_rate 4e-2 \
    --pretrained_learning_rate 4e-3 \
    --momentum 0.9 \
    --num_epochs 100 \
    --augment True

# python train_from_pretrained_dataparallel.py \
#     --model resnet18 \
#     --pretrained \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-3 \
#     --momentum 0.9 \
#     --num_epochs 100 \
#     --weight_decay 1e-3

# dropout 0.5
# python train_from_pretrained_dataparallel.py \
#     --model resnet18 \
#     --pretrained \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-3 \
#     --momentum 0.9 \
#     --num_epochs 100 \
#     --dropout_rate 0.5 \
#     --weight_decay 1e-3 \
#     --dropout_rate 0.5

# augment
# python train_from_pretrained_dataparallel.py \
#     --model resnet18 \
#     --pretrained \
#     --data_dir /share/home/zjy/data/CUB_200_2011 \
#     --batch_size 512 \
#     --fc_learning_rate 4e-2 \
#     --pretrained_learning_rate 4e-3 \
#     --momentum 0.9 \
#     --num_epochs 100 \
#     --dropout_rate 0.5 \
#     --weight_decay 1e-3 \
#     --augment \
#     --dropout_rate 0

python train_from_pretrained_dataparallel.py \
    --model resnet18 \
    --pretrained \
    --data_dir /share/home/zjy/data/CUB_200_2011 \
    --batch_size 512 \
    --fc_learning_rate 4e-2 \
    --pretrained_learning_rate 4e-3 \
    --momentum 0.9 \
    --scheduler StepLR \
    --step_size 30 \
    --num_epochs 120 \
    --dropout_rate 0 \
    --weight_decay 1e-3 \
    --dropout_rate 0 \
    --scheduler StepLR

python train_from_pretrained_dataparallel.py \
    --model resnet50 \
    --pretrained \
    --data_dir /share/home/zjy/data/CUB_200_2011 \
    --batch_size 512 \
    --fc_learning_rate 2e-2 \
    --pretrained_learning_rate 2e-3 \
    --momentum 0.9 \
    --scheduler StepLR \
    --step_size 20 \
    --num_epochs 200 \
    --dropout_rate 0 \
    --weight_decay 1e-3 \
    --dropout_rate 0





