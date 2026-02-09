# utils/common.py
import os
import numpy as np
import torch


def mkdir_for(file_path):
    """创建文件路径所需的目录"""
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)


def model_size(model):
    """计算模型参数数量"""
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def torch_set_gpu(gpus):
    """设置GPU设备"""
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        print('Launching on CPU')

    return cuda


