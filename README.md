# SimMatchV2: Semi-Supervised Learning with Graph Consistency (ICCV2023)

This repository contains PyTorch evaluation code, training code and pretrained models for SimMatchV2. Most of the code in this repository is adapt from [here](https://github.com/mingkai-zheng/simmatch/tree/main).

For details see [SimMatchV2: Semi-Supervised Learning with Graph Consistency](https://arxiv.org/abs/2308.06692) by Mingkai Zheng, Shan You, Lang Huang, Chen Luo, Fei Wang, Chen Qian, and Chang Xu

## Reproducing
To run the code, you probably need to change the Dataset setting (ImagenetPercent function in [data/imagenet.py](data/imagenet.py)), and Pytorch DDP setting (dist_init function in [utils/dist_utils.py](utils/dist_utils.py)) for your server environment.

The distributed training of this code is based on slurm environment, we have provided the training scrips in script/train.sh

We also provide the pre-trained model. 

|          |Arch | Setting | Epochs  | Accuracy | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|
|  SimMatchV2 | ResNet50 | 1% | 300  | 71.9 % | [300ep-res50-1p.pth](https://drive.google.com/file/d/1ZAt0ppIuhwbuY7S-cEVKocMbv5m75QiL/view?usp=sharing) |
|  SimMatchV2 | ResNet50 | 10% | 300  | 76.2 % | [300ep-res50-10p.pth](https://drive.google.com/file/d/1L3l0nJ8owz71zkN2BHS-Bv_LXFJMxsBR/view?usp=sharing) |

If you want to test the pre-trained model, please download the weights from the link above, and move them to the [checkpoints](checkpoints) folder. The evaluation scripts also have been provided in [script/train.sh](script/train.sh)


## Citation
If you find that SimMatch interesting and help your research, please consider citing it:
```
@misc{zheng2023simmatchv2,
      title={SimMatchV2: Semi-Supervised Learning with Graph Consistency}, 
      author={Mingkai Zheng and Shan You and Lang Huang and Chen Luo and Fei Wang and Chen Qian and Chang Xu},
      year={2023},
      eprint={2308.06692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
