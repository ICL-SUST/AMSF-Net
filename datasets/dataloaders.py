import os
import math
import torch
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
from . import samplers, transform_manager


def get_dataset(data_path, is_training, transform_type, pre, use_dwt=False,
                concat_rgb=True, use_multi_scale=False, dwt_levels=3):
    if pre is None:
        pre = False

    dataset = datasets.ImageFolder(
        data_path,
        loader=lambda x: image_loader(
            path=x,
            is_training=is_training,
            transform_type=transform_type,
            pre=pre,
            use_dwt=use_dwt,
            concat_rgb=concat_rgb,
            use_multi_scale=use_multi_scale,
            dwt_levels=dwt_levels
        )
    )

    return dataset


def meta_train_dataloader(data_path, way, shots, transform_type, use_dwt=False,
                          concat_rgb=True, use_multi_scale=False, dwt_levels=3, trial=1000):
    dataset = get_dataset(
        data_path=data_path,
        is_training=True,
        transform_type=transform_type,
        pre=False,
        use_dwt=use_dwt,
        concat_rgb=concat_rgb,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=samplers.meta_batchsampler(data_source=dataset, way=way, shots=shots, trial=trial),
        num_workers=0,
        pin_memory=False)

    return loader


def meta_test_dataloader(data_path, way, shot, pre, transform_type=None, query_shot=16,
                         use_dwt=False, concat_rgb=True, use_multi_scale=False,
                         dwt_levels=3, trial=1000):
    if pre is None:
        pre = False

    dataset = get_dataset(
        data_path=data_path,
        is_training=False,
        transform_type=transform_type,
        pre=pre,
        use_dwt=use_dwt,
        concat_rgb=concat_rgb,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=samplers.random_sampler(
            data_source=dataset,
            way=way,
            shot=shot,
            query_shot=query_shot,
            trial=trial
        ),
        num_workers=0,
        pin_memory=False)

    return loader


def normal_train_dataloader(data_path, batch_size, transform_type, use_dwt=False,
                            concat_rgb=True, use_multi_scale=False, dwt_levels=3):
    dataset = get_dataset(
        data_path=data_path,
        is_training=True,
        transform_type=transform_type,
        pre=False,
        use_dwt=use_dwt,
        concat_rgb=concat_rgb,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True)

    return loader


def image_loader(path, is_training, transform_type, pre, use_dwt=False,
                 concat_rgb=True, use_multi_scale=False, dwt_levels=3):

    p = Image.open(path)

    if p.mode == 'RGBA':
        p = p.convert('RGB').convert('L')
    elif p.mode == 'RGB':
        p = p.convert('L')
    elif p.mode != 'L':
        p = p.convert('L')

    final_transform = transform_manager.get_transform(
        is_training=is_training,
        transform_type=transform_type,
        pre=pre,
        use_dwt=use_dwt,
        concat_rgb=concat_rgb,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    p = final_transform(p)

    return p


def verify_data_format_multiscale(data_path, use_multi_scale=True, dwt_levels=3):

    import random

    dataset = datasets.ImageFolder(data_path)
    sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    for idx in sample_indices:
        img_path, label = dataset.imgs[idx]
        img = Image.open(img_path)
        print(f"Image {idx}: path={img_path}, mode={img.mode}, size={img.size}, label={label}")

        if img.mode != 'L':
            img = img.convert('L')
        img_array = np.array(img)
        print(f"  Stats: min={img_array.min()}, max={img_array.max()}, "
              f"mean={img_array.mean():.2f}, std={img_array.std():.2f}")

        if use_multi_scale:
            from .transform_manager import get_transform
            transform = get_transform(
                is_training=False,
                transform_type=0,
                pre=False,
                use_dwt=True,
                concat_rgb=True,
                use_multi_scale=True,
                dwt_levels=dwt_levels
            )

            output = transform(img)
            print(f"  Multi-scale output shape: {output.shape}")  # Should be [1+3L, H, W]
            print(f"  Output channels: LL + {dwt_levels} x (LH, HL, HH) = {1 + 3 * dwt_levels} channels")



def collate_mri_batch(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.LongTensor(targets)
    return images, targets


def create_mri_dataloader_multiscale(data_path, batch_size, is_training=True,
                                     use_multi_scale=True, dwt_levels=3,
                                     num_workers=0, pin_memory=False):
    dataset = get_dataset(
        data_path=data_path,
        is_training=is_training,
        transform_type=0,
        pre=False,
        use_dwt=True,
        concat_rgb=True,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
        collate_fn=collate_mri_batch
    )

    return dataloader