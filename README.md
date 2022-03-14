# ViT-P PyTorch implementation

This is a PyTorch/GPU re-implementation of the paper: [ViT-P: Rethinking Data-efficient Vision Transformers from Locality](https://arxiv.org/abs/2203.02358).

## TODO

- [ ] run script
- [ ] model weights

## Usage

```shell
# training vitp_tiny on cifar100
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=1 main_scratch.py \
    --accum_iter 1 --batch_size 256 \
    --num_workers 8 \
    --model vitp_tiny_patch2_32 --input_size 32 \
    --epochs 300 --blr 5e-4 --layer_decay 1.0 \
    --warmup_epochs 20 --weight_decay 0.3 --drop_path 0.1 --opt_betas 0.9 0.95 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 --color_jitter 0.4 \
    --dist_eval --dataset cifar100 --data_path ./data --nb_classes 100 \
    --output_dir output/vitp_tiny_patch2_cifar100 \
    --global_pool --mean 0.5070 0.4865 0.4409 --std 0.2673 0.2564 0.2761 \
> info.log 2>&1 &

# training vitp_tiny_patch16_224 on imagenet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=1 \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 main_scratch.py \
    --accum_iter 1 --batch_size 512 \
    --num_workers 8 \
    --model vitp_tiny_patch16_224 --input_size 224 \
    --epochs 300 --blr 1e-4 --layer_decay 1.0 \
    --warmup_epochs 20 --weight_decay 0.3 --drop_path 0.1 --opt_betas 0.9 0.95 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 --color_jitter 0.4 \
    --dist_eval --dataset imagenet --data_path ./data/ILSVRC-2012/ --nb_classes 1000 \
    --output_dir output/vitp_tiny_patch16_imagenet \
    --global_pool --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
> info.log 2>&1 &

# training vit_tiny on cifar100
# our results are in the BASE model, the difference being the use of 64 mini-batch.
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=1 main_scratch.py \
    --accum_iter 1 --batch_size 256 \
    --num_workers 8 \
    --model vit_tiny_patch2 --input_size 32 \
    --epochs 300 --blr 5e-4 --layer_decay 1.0 \
    --warmup_epochs 20 --weight_decay 0.3 --drop_path 0.1 --opt_betas 0.9 0.95 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 --color_jitter 0.4 \
    --dist_eval --dataset cifar100 --data_path ./data --nb_classes 100 \
    --output_dir output/vit_tiny_patch2_cifar100 \
    --global_pool --mean 0.5070 0.4865 0.4409 --std 0.2673 0.2564 0.2761 \
> info.log 2>&1 &
```


