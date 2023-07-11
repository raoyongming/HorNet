# HorNet <img width="32" alt="HorNet Icon" src="figs/hornet_icon.png">

Created by [Yongming Rao](https://raoyongming.github.io/)\*, [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Yansong Tang](https://andytang15.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Ser-Nam Lim](https://sites.google.com/site/sernam)†, [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)†

This repository contains PyTorch implementation for HorNet (NeurIPS 2022).

HorNet is a family of generic vision backbones that perform explicit *high-order* spatial interactions based on Recursive Gated Convolution.

![intro](figs/intro.jpg)

[[Project Page]](https://hornet.ivg-research.xyz/) [[arXiv]](https://arxiv.org/abs/2207.14284)

## Model Zoo

ImageNet-1K trained models:
| name | arch | Params | FLOPs | Top-1 | url |
| --- | --- | --- | --- | --- |  --- |
| HorNet-T (7x7) | ```hornet_tiny_7x7``` | 22M | 4.0G | 82.8 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/1ca970586c6043709a3f/?dl=1)|
| HorNet-T (GF) | ```hornet_tiny_gf``` | 23M | 3.9G | 83.0 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/511faad0bde94dfcaa54/?dl=1)|
| HorNet-S (7x7) | ```hornet_small_7x7``` | 50M | 8.8G | 83.8 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/46422799db2941f7b684/?dl=1)|
| HorNet-S (GF) | ```hornet_small_gf``` | 50M | 8.7G | 84.0 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8405c984bf084d2ba85a/?dl=1)|
| HorNet-B (7x7) | ```hornet_base_7x7``` | 87M | 15.6G | 84.2 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/5c86cb3d655d4c17a959/?dl=1)|
| HorNet-B (GF) | ```hornet_base_gf``` | 88M | 15.5G | 84.3 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6c84935e63b547f383fb/?dl=1)|

ImageNet-22K trained models:
| name | arch | Params | FLOPs | url |
| --- | --- | --- | --- | --- |
| HorNet-L (7x7) | ```hornet_large_7x7``` | 209M | 34.8G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a76f7ce114ad4e9c8ffb/?dl=1)|
| HorNet-L (GF) | ```hornet_large_gf``` | 211M | 34.7G |  [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/12c99a79e77341e59681/?dl=1)|
| HorNet-L (GF)* | ```hornet_large_gf_img384``` | 216M | 101.8G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/9253bd0d50c84ab1b71e/?dl=1)|

*indicate the model is finetuned to 384x384 resolution on ImageNet-22k.

## ImageNet Classification

### Requirements

- torch==1.8.0
- torchvision==0.9.0
- timm==0.4.12
- tensorboardX 
- six
- submitit (multi-node training)

**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation

To evaluate a pre-trained HorNet model on the ImageNet validation set with 8 GPUs, run:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model hornet_tiny_7x7 --eval true --input_size 224 \
--resume /path/to/checkpoint \ 
--data_path /path/to/imagenet-1k
```

### Training

To train HorNet models on ImageNet from scratch on a single machine, run:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model hornet_tiny_7x7 --drop_path 0.2 --clip_grad 5\
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir ./logs/hornet_tiny_7x7
```

We provide detailed training commands for our models in [TRAINING.md](./TRAINING.md).

## Downstream Tasks

Please check the [object_detection.md](./object_detection/README.md) and [semantic_segmentation.md](./semantic_segmentation/README.md) for training and evaluation instructions on dense prediction tasks.

HorNet also achieves state-of-the-art performance on 3D object classification with our new framework ([***P2P***](https://github.com/wangzy22/P2P)) to leverage pre-trained image models for point cloud understanding.

## License
MIT License

## Acknowledgements
Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). We would like to thank [High-Flyer AI Research](https://www.high-flyer.cn/) for their generous support of partial computational resources used in this project.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{rao2022hornet,
  title={HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions},
  author={Rao, Yongming and Zhao, Wenliang and Tang, Yansong and Zhou, Jie and Lim, Ser-Lam and Lu, Jiwen},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
