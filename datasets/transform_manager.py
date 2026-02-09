import torch
import torchvision.transforms as transforms
from .dwt_transform import WFDTTransform, MultiScaleWFDTTransform


class MultiScaleWFDTWrapper:

    def __init__(self, levels=3, normalize_transform=None):
        self.wfdt = MultiScaleWFDTTransform(wavelet='db4', levels=levels)
        self.to_tensor = transforms.ToTensor()
        self.normalize = normalize_transform
        self.levels = levels

    def __call__(self, img):
        if img.mode == 'RGB':
            img = img.convert('L')

        multi_scale_features = self.wfdt(img)

        features_tensor = torch.from_numpy(multi_scale_features).float() / 255.0

        if self.normalize:
            for i in range(features_tensor.shape[0]):
                channel = features_tensor[i:i + 1]
                features_tensor[i] = (channel - channel.mean()) / (channel.std() + 1e-8)

        return features_tensor


class WFDTWrapper:

    def __init__(self, normalize_transform=None):

        self.wfdt = WFDTTransform()
        self.to_tensor = transforms.ToTensor()
        self.normalize = normalize_transform

    def __call__(self, img):
        if img.mode == 'RGB':
            img = img.convert('L')

        high_freq_img = self.wfdt(img)

        original_tensor = self.to_tensor(img)
        high_freq_tensor = self.to_tensor(high_freq_img)

        combined_tensor = torch.cat([original_tensor, high_freq_tensor], dim=0)

        if self.normalize:
            combined_tensor = (combined_tensor - combined_tensor.mean()) / combined_tensor.std()

        return combined_tensor


def get_transform(is_training=None, transform_type=None, pre=None, use_dwt=False,
                  concat_rgb=True, use_multi_scale=False, dwt_levels=3):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    normalize = None

    if pre:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    if is_training:
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]

        if use_dwt and concat_rgb:
            if use_multi_scale:
                transform_list.append(MultiScaleWFDTWrapper(levels=dwt_levels, normalize_transform=normalize))
            else:
                transform_list.append(WFDTWrapper(normalize))
        else:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        return transforms.Compose(transform_list)

    # 评估模式
    else:
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        if use_dwt and concat_rgb:
            if use_multi_scale:
                transform_list.append(MultiScaleWFDTWrapper(levels=dwt_levels, normalize_transform=normalize))
            else:
                transform_list.append(WFDTWrapper(normalize))
        else:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        return transforms.Compose(transform_list)