#!/bin/bash


# ======= 1% setting ==========
# ./script/command.sh 300ep-res50-1p   partition 8 1 "python3 -u main.py  --norm_feat --multicrop --alpha 0.1 --topn 128 --k 256 --lambda_nn 10 --lambda_ee 10 --lambda_ne 10  --DA --da_m 0.9 --t 0.1 --fp16 --nesterov  --lr 0.03 --epochs 300 --cos --warmup-epoch 5 --anno-percent 0.01 --checkpoint checkpoints/300ep-res50-1p.pth "

# ======= 10% setting ==========
# ./script/command.sh 300ep-res50-10p  partition 8 1 "python3 -u main.py  --norm_feat --multicrop --alpha 0.1 --topn 128 --k 256 --lambda_nn 10 --lambda_ee 5  --lambda_ne 5   --DA --da_m 0.9 --t 0.1 --fp16 --nesterov  --lr 0.03 --epochs 300 --cos --warmup-epoch 5 --anno-percent 0.1  --checkpoint checkpoints/300ep-res50-10p.pth"







# ======= evaluate 1% ==========
# ./script/command.sh eval-300ep-res50-1p  partition  8 1 "python3 -u main.py --evaluate --anno-percent 0.01  --norm_feat --k 256 --checkpoint checkpoints/300ep-res50-1p.pth"
# * Acc@1 76.246 Acc@5 92.222

# ======= evaluate 10% ==========
# ./script/command.sh eval-300ep-res50-10p  partition 8 1 "python3 -u main.py --evaluate --anno-percent 0.1   --norm_feat --k 256 --checkpoint checkpoints/300ep-res50-10p.pth"
# * Acc@1 71.920 Acc@5 89.978