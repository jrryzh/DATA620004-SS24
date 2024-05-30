# train
python code/plain_train_net.py --config-file configs/faster_rcnn_R_101_FPN_3x.yaml --num-gpus 8
# eval
python code/plain_train_net.py --config-file configs/faster_rcnn_R_101_FPN_3x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS output_2007/model_0089999.pth