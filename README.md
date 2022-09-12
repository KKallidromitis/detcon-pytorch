# detcon-pytorch

PyTorch implementation of DetconB: Efficient Visual Pretraining with Contrastive Detection.

Installation of [Apex](https://github.com/NVIDIA/apex) is required to enable DDP.

To log metrics to [wandb](https://github.com/wandb/client) switch to `enable_wandb:True` in `train_imagenet_300.yaml`

## Requirements

```
python>=3.9
pytorch>=1.10.0
pyyaml
tensorboardx
scikit-learn
scikit-image
opencv-python
pillow-simd
wandb
```

This repo uses `torch.distributed.launch` for pretraining:

```bash
python -m torch.distributed.launch --nproc_per_node=4--nnodes=32 --node_rank=0 --master_addr="" --master_port=12345 detconb_main.py {CONFIG_FILENAME}
```

## Dataset Structure

```none
imagenet
├── images
│   ├── train
│   │   ├── n01440764
│   │   ├── ...
│   │   ├── n15075141
│   ├── val
│   │   ├── n01440764
│   │   ├── ...
│   │   ├── n15075141
```

## Pretrained Weights

We release pretrained weights pretrained on ImageNet-1k for 300 epochs in torchvision format.

[Download](https://drive.google.com/file/d/15a7jJ1XVmSVZVo0xFE4gDn1Uw2Mns9Ui/view?usp=sharing) 

The evaluation baselines are as follows

|         Metric         | Value  |
|------------------|---|
|  PASCAL VOC mIoU | 76.0 |
| Cityscapes mIoU  | 76.2  |
|    MS COCO $\text{AP}^{\text{bb}}$ | 41.5  |
|    MS COCO $\text{AP}^{\text{mk}}$ |  38.3 |

## Reproduce Results

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for PASCAL VOC and Cityscapes semantic segmentation. We use [detectron2](https://github.com/facebookresearch/detectron2) for MS COCO object detection and instance segmentation. The corresponding config can be found in `eval` folder.

## Acknowledgement

This repo is based on the BYOL implementation from Yao: https://github.com/yaox12/BYOL-PyTorch and K-Means implementation from Ali Hassani https://github.com/alihassanijr/TorchKMeans