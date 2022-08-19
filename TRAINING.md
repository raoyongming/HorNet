# Training

We provide ImageNet-1K training, ImageNet-22K pre-training, and ImageNet-1K fine-tuning commands here.

## ImageNet-1K Training 

HorNet-T (7x7)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_tiny_7x7 --drop_path 0.2 --clip_grad 100 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_tiny_7x7
```

HorNet-T (GF)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_tiny_gf --drop_path 0.2 --clip_grad 1 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_tiny_gf
```

HorNet-S (7x7)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_small_7x7 --drop_path 0.4 --clip_grad 5 \
--batch_size 64 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_small_7x7
```

HorNet-S (GF)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_small_gf --drop_path 0.4 --clip_grad 1 \
--batch_size 64 --lr 4e-3 --update_freq 2 --aa rand-m12-mstd0.5-inc1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_small_gf
```

HorNet-B (7x7)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_base_7x7 --drop_path 0.5 --clip_grad 5 \
--batch_size 64 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_base_7x7
```

HorNet-B (GF)
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_base_gf --drop_path 0.5 --clip_grad 1 \
--batch_size 64 --lr 4e-3 --update_freq 2 --aa rand-m15-mstd0.5-inc1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_base_gf
```


## ImageNet-22K Pre-Training 

HorNet-L (7x7)
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model hornet_large_7x7 --drop_path 0.2 --clip_grad 5 --weight_decay 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 2 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder  --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir ./logs/hornet_large_7x7_in22k
```

HorNet-L (GF)
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model hornet_large_gf --drop_path 0.2 --clip_grad 1 --weight_decay 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 2 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder  --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir ./logs/hornet_large_gf_in22k
```

## Finetuning ImageNet-22K Models to ImageNet-1K

HorNet-L (7x7) @ 384x384
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_large_7x7 --drop_path 0.4 --clip_grad 1 \
--batch_size 16 --lr 5e-5 --update_freq 1 --weight_decay 1e-8 \
--warmup_epochs 0 --input_size 384 --epochs 30 \
--model_ema true --model_ema_eval true --cutmix 0 --mixup 0 \
--layer_decay 0.7 --finetune ./pretrained/hornet_large_7x7_in22k.pth \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_larget_7x7_img384_ft1k
```

HorNet-L (GF) @ 384x384
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model hornet_large_gf_img384 --drop_path 0.4 --clip_grad 1 \
--batch_size 16 --lr 5e-5 --update_freq 1 --weight_decay 1e-8 \
--warmup_epochs 0 --input_size 384 --epochs 30 \
--model_ema true --model_ema_eval true --cutmix 0 --mixup 0 \
--layer_decay 0.7 --finetune ./pretrained/hornet_large_gf_in22k.pth \
--data_path /path/to/imagenet-1k \
--job_dir ./logs/hornet_larget_gf_img384_ft1k
```
