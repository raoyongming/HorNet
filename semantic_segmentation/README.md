# ADE20k Semantic segmentation with HorNet


## Results and Fine-tuned Models

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|
| HorNet-T-7x7 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/762f05c3c8cd4743b534/?dl=1) | UPerNet | 512x512 | 160K | 48.1 | 48.9 | 52M | 926G | [model]() |
| HorNet-T-GF | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/395dd6c443ed4a339739/?dl=1) | UPerNet | 512x512 | 160K | 49.2 | 49.3 | 55M | 924G | [model]() |
| HorNet-S-7x7 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/9d7043023da14e4b8b2e/?dl=1) | UPerNet | 512x512 | 160K | 49.2 | 49.8 | 81M | 1030G | [model]() |
| HorNet-S-GF | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/19eef725b2e64692b8b0/?dl=1) | UPerNet | 512x512 | 160K | 50.0 | 50.5 | 85M | 1027G | [model]() |
| HorNet-B-7x7 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/836ab04898c646c389ce/?dl=1) | UPerNet | 512x512 | 160K | 50.0 | 50.5 | 121M | 1174G | [model](h) |
| HorNet-B-GF | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/60f706e36f6b4098a1f9/?dl=1) | UPerNet | 640x640 | 160K | 50.5 | 50.9 | 126M | 1171G | [model]() |
| HorNet-L-7x7 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4de41e26cb254c28a61a/?dl=1) | UPerNet | 640x640 | 160K | 54.1 | 54.5 | 232M | 2473G | [model]() |
| HorNet-L-GF | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/f36957d46eef47da9c25/?dl=1) | UPerNet | 640x640 | 160K | 55.0 | 55.2 | 239M | 2465G | [model]() |

### Training

```
bash dist_train.sh <CONFIG_PATH> <NUM_GPUS> --work-dir <SAVE_PATH> --options model.pretrained=<PRETRAIN_MODEL>
```

For example, using a `HorNet-T-GF` backbone with UperNet:
```bash
bash dist_train.sh \
    configs/hornet/upernet_hornet_tiny_gf_512_160k_ade20k.py 8 \
    --work-dir /path/to/save \
    --options model.pretrained=/path/to/pretrained/weight
```

More config files can be found at [`configs/convnext`](configs/convnext).


## Evaluation
Note: Please add `from backbone import convnext` to tools/test.py.

Command format for multi-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a `ConvNeXt-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ms.py \ 
    https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth 4 --eval mIoU --aug-test
```

Command format for single-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a `ConvNeXt-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ss.py \ 
    https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth 4 --eval mIoU
```

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [XCiT](https://github.com/facebookresearch/xcit), [SETR](https://github.com/fudan-zvg/SETR) repositories.
