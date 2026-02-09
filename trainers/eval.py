import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

sys.path.append('..')
from datasets import dataloaders
from tqdm import tqdm


def get_score(acc_list):
    mean = np.mean(acc_list)
    interval = 1.96 * np.sqrt(np.var(acc_list) / len(acc_list))
    return mean, interval


def meta_test(data_path, model, way, shot, pre, transform_type, query_shot=16,
              trial=10000, return_list=False, use_dwt=False, use_multi_scale=False, dwt_levels=3):


    # 创建数据加载器，支持多尺度DWT
    eval_loader = dataloaders.meta_test_dataloader(
        data_path=data_path,
        way=way,
        shot=shot,
        pre=pre,
        transform_type=transform_type,
        query_shot=query_shot,
        trial=trial,
        use_dwt=use_dwt,
        concat_rgb=True,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels
    )

    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    acc_list = []

    start_time = time.time()

    for i, (inp, _) in enumerate(eval_loader):
        inp = inp.cuda()
        max_index = model.meta_test(inp, way=way, shot=shot, query_shot=query_shot)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
        acc_list.append(acc)


        progress = (i + 1) / trial
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        elapsed_time = time.time() - start_time
        if i > 0:
            avg_time_per_iter = elapsed_time / (i + 1)
            remaining_time = avg_time_per_iter * (trial - i - 1)
            time_str = f'{elapsed_time:.0f}s<{remaining_time:.0f}s'
        else:
            time_str = 'calculating...'

        sys.stdout.write(f'\rValidating: {progress * 100:3.0f}%|{bar}| {i + 1}/{trial} [{time_str}]')
        sys.stdout.flush()

    print()

    if return_list:
        return np.array(acc_list)
    else:
        mean, interval = get_score(acc_list)
        return mean, interval