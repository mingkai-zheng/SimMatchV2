# SimMatchV2: Semi-Supervised Learning with Graph Consistency (ICCV2023)

This repository contains PyTorch evaluation code, training code and pretrained models for SimMatchV2. Most of the code in this repository is adapt from [here](https://github.com/mingkai-zheng/simmatch/tree/main).

<!-- For details see [SimMatch: Semi-supervised Learning with Similarity Matching](https://arxiv.org/abs/2203.06915) by Mingkai Zheng, Shan You, Fei Wang, Chen Qian, and Chang Xu -->

## Reproducing
To run the code, you probably need to change the Dataset setting (ImagenetPercent function in [data/imagenet.py](data/imagenet.py)), and Pytorch DDP setting (dist_init function in [utils/dist_utils.py](utils/dist_utils.py)) for your server environment.

The distributed training of this code is based on slurm environment, we have provided the training scrips in script/train.sh

We also provide the pre-trained model. 

|          |Arch | Setting | Epochs  | Accuracy | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|
|  SimMatchV2 | ResNet50 | 1% | 300  | 71.9 % | [300ep-res50-1p.pth](https://drive.google.com/file/d/1N-i7QwAyUuc862jm_nZLCKJL2cJCvbbD/view?usp=sharing) |
|  SimMatchV2 | ResNet50 | 10% | 300  | 76.2 % | [300ep-res50-10p.pth](https://drive.google.com/file/d/1Eeeqxixr9JtbrUmFDgRcf-tCWbPGnt2o/view?usp=sharing) |

If you want to test the pre-trained model, please download the weights from the link above, and move them to the checkpoints folder. The evaluation scripts also have been provided in script/train.sh

<!-- 
## Citation
If you find that SimMatch interesting and help your research, please consider citing it:
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Mingkai and You, Shan and Huang, Lang and Wang, Fei and Qian, Chen and Xu, Chang},
    title     = {SimMatch: Semi-Supervised Learning With Similarity Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14471-14481}
}
``` -->
