# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from turtle import down
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == 'cifar10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(train=is_train, root=args.data_path, download=True)
        dataset.transform = transform
    elif args.dataset == 'cifar100':
        from torchvision.datasets import CIFAR100
        dataset = CIFAR100(train=is_train, root=args.data_path, download=True)
        dataset.transform = transform
    else:
        raise ValueError('Dataset name is implement: {}'.format(args.datset))
    return dataset


def build_transform(is_train, args):
    mean = args.mean or IMAGENET_INCEPTION_MEAN  # mean = IMAGENET_DEFAULT_MEAN
    std = args.std or IMAGENET_INCEPTION_STD  # std = IMAGENET_DEFAULT_STD
    
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=args.scale,
            ratio=args.ratio,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if 'cifar' in args.dataset and args.input_size == 32:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    # eval transform
    if 'cifar' in args.dataset and args.input_size == 32:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std), 
        ])
    else:
        crop_pct = 224 / 256
        size = int(args.input_size / crop_pct)
        return transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std), 
        ])
