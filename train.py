import os
import sys
import torch
import yaml
from functools import partial
import torchvision
import argparse

sys.path.append('../../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
from models.FRN import FRN, create_frn_with_multiscale_dwt_vit


def load_model_safely(model, checkpoint_path, strict=False):

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model_dict = model.state_dict()

    filtered_dict = {}
    skipped_keys = []

    for key, value in checkpoint.items():
        if key in model_dict:
            if 'cached_' in key:
                print(f"跳过缓存buffer '{key}'")
                skipped_keys.append(key)
                continue
            elif model_dict[key].shape == value.shape:
                filtered_dict[key] = value
            else:
                if not strict:
                    print(f"警告: 跳过 '{key}' 由于形状不匹配: "
                          f"检查点 {value.shape} vs 模型 {model_dict[key].shape}")
                    skipped_keys.append(key)
                else:
                    raise RuntimeError(f"形状不匹配 '{key}': "
                                       f"检查点 {value.shape} vs 模型 {model_dict[key].shape}")
        else:
            if not strict:
                print(f"警告: 键 '{key}' 在当前模型中未找到")
            else:
                raise RuntimeError(f"键 '{key}' 在当前模型中未找到")

    model_dict.update(filtered_dict)

    model.load_state_dict(model_dict, strict=False)

    print(f"\n成功加载检查点!")
    if skipped_keys:
        print(f"跳过了 {len(skipped_keys)} 个不匹配的键")

    return model


# 数据检查函数
def check_data_loading(pm, args, use_multi_scale=False, dwt_levels=3):
    print("\n=== 检查数据加载 ===")

    test_loader = dataloaders.meta_train_dataloader(
        data_path=pm.train,
        way=4,
        shots=[5, 15],
        transform_type=args.train_transform_type,
        use_dwt=True,
        concat_rgb=True,
        use_multi_scale=use_multi_scale,
        dwt_levels=dwt_levels,
        trial=1  # 只加载一个batch
    )

    for batch_idx, (data, labels) in enumerate(test_loader):
        if use_multi_scale:
            expected_channels = 1 + 3 * dwt_levels
            print(f"多尺度模式 - 期望通道数: {expected_channels}")
        else:
            expected_channels = 2
            print(f"标准模式 - 期望通道数: {expected_channels}")

        print(f"Batch shape: {data.shape}")  # 应该是 [80, expected_channels, 224, 224]
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

        if use_multi_scale:
            print(f"LL channel range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}]")
            for l in range(dwt_levels):
                base_idx = 1 + l * 3
                print(f"Level {l + 1} - LH: [{data[:, base_idx].min():.3f}, {data[:, base_idx].max():.3f}]")
                print(f"Level {l + 1} - HL: [{data[:, base_idx + 1].min():.3f}, {data[:, base_idx + 1].max():.3f}]")
                print(f"Level {l + 1} - HH: [{data[:, base_idx + 2].min():.3f}, {data[:, base_idx + 2].max():.3f}]")
        else:
            print(f"Channel 0 (原图) range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}]")
            print(f"Channel 1 (高频) range: [{data[:, 1].min():.3f}, {data[:, 1].max():.3f}]")

        os.makedirs('debug_images', exist_ok=True)

        if use_multi_scale:
            torchvision.utils.save_image(data[:8, 0:1], 'debug_images/LL_channel.png', normalize=True)
            for l in range(dwt_levels):
                base_idx = 1 + l * 3
                torchvision.utils.save_image(data[:8, base_idx:base_idx + 1], f'debug_images/L{l + 1}_LH.png',
                                             normalize=True)
                torchvision.utils.save_image(data[:8, base_idx + 1:base_idx + 2], f'debug_images/L{l + 1}_HL.png',
                                             normalize=True)
                torchvision.utils.save_image(data[:8, base_idx + 2:base_idx + 3], f'debug_images/L{l + 1}_HH.png',
                                             normalize=True)
        else:
            torchvision.utils.save_image(data[:8, 0:1], 'debug_images/original_channel.png', normalize=True)
            torchvision.utils.save_image(data[:8, 1:2], 'debug_images/dwt_channel.png', normalize=True)

        print("调试图像已保存到 debug_images/ 目录")
        break

    print("=== 数据加载检查完成 ===\n")


def add_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vit_model_name",
                        help="timm ViT model name",
                        type=str,
                        default='vit_base_patch16_224.augreg_in21k',
                        choices=['vit_small_patch16_224',
                                 'vit_base_patch16_224',
                                 'vit_large_patch16_224',
                                 'vit_base_patch16_224.augreg_in21k',
                                 'vit_base_patch16_384',
                                 'vit_base_patch32_224_in21k'])

    parser.add_argument("--use_multi_scale",
                        help="use multi-scale DWT",
                        action="store_true",
                        default=True)

    parser.add_argument("--dwt_levels",
                        help="number of DWT decomposition levels",
                        type=int,
                        default=3)

    parser.add_argument("--skip_data_check",
                        help="skip data loading check",
                        action="store_true",
                        default=False)

    parser.add_argument("--warmup_epochs",
                        help="number of warmup epochs",
                        type=int,
                        default=300)

    parser.add_argument("--drop_rate",
                        help="dropout rate",
                        type=float,
                        default=0.1)

    parser.add_argument("--attn_drop_rate",
                        help="attention dropout rate",
                        type=float,
                        default=0.05)

    parser.add_argument("--drop_path_rate",
                        help="drop path rate",
                        type=float,
                        default=0.15)

    parser.add_argument("--sparsity_lambda",
                        help="sparsity regularization weight for directional gating",
                        type=float,
                        default=0.0)  # 默认不使用

    parser.add_argument("--sparsity_mu",
                        help="sparsity regularization weight for scale gating",
                        type=float,
                        default=0.0)  # 默认不使用
    parser.add_argument("--use_context_refinement",
                        help="use context refinement in multi-scale DWT-ViT",
                        action="store_true",
                        default=True)  # 默认启用
    return parser


def compute_sparsity_loss(model, lambda_dir=0.01, mu_scale=0.01):

    if not hasattr(model, 'dwt_vit') or not hasattr(model.dwt_vit, 'cached_band_weights'):
        return 0

    sparsity_loss = 0


    if model.dwt_vit.cached_band_weights is not None:
        band_weights = model.dwt_vit.cached_band_weights

        entropy_dir = -torch.sum(band_weights * torch.log(band_weights + 1e-8), dim=-1)
        sparsity_loss += lambda_dir * entropy_dir.mean()


    if model.dwt_vit.cached_scale_weights is not None:
        scale_weights = model.dwt_vit.cached_scale_weights

        entropy_scale = -torch.sum(scale_weights * torch.log(scale_weights + 1e-8), dim=-1)
        sparsity_loss += mu_scale * entropy_scale.mean()

    return sparsity_loss


def main():
    args = trainer.train_parser()


    custom_parser = add_train_args()
    custom_args, _ = custom_parser.parse_known_args()

    # 合并参数
    for arg in vars(custom_args):
        setattr(args, arg, getattr(custom_args, arg))

    # 修改训练参数
    print("\n=== 训练参数设置 ===")

    # 根据模型调整学习率
    if 'large' in args.vit_model_name:
        args.lr = 5e-5  # 大模型使用更小的学习率
    elif 'base' in args.vit_model_name:
        args.lr = 2e-5
    else:  # small
        args.lr = 1.5e-5

    print(f"模型: {args.vit_model_name}")
    print(f"多尺度DWT: {args.use_multi_scale}")
    if args.use_multi_scale:
        print(f"DWT分解层数: {args.dwt_levels}")
    print(f"学习率: {args.lr}")

    # 设置学习率衰减点
    if args.decay_epoch is None:
        args.decay_epoch = [1500, 2500, 3500, 4500, 5500]
    print(f"学习率衰减点: {args.decay_epoch}")

    # Dropout设置
    print(f"Dropout率: {args.drop_rate}")
    print(f"Attention Dropout率: {args.attn_drop_rate}")
    print(f"Drop Path率: {args.drop_path_rate}")

    # 正则化设置
    if args.sparsity_lambda > 0 or args.sparsity_mu > 0:
        print(f"稀疏性正则化 - 方向门控: {args.sparsity_lambda}, 尺度门控: {args.sparsity_mu}")

    # 读取配置文件
    with open('./config.yml', 'r', encoding='utf-8') as f:
        temp = yaml.safe_load(f)
    data_path = os.path.abspath(temp['data_path'])

    fewshot_path = os.path.join(data_path, 'COVID')  # MRI数据路径

    pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)

    if not args.skip_data_check:
        check_data_loading(pm, args, use_multi_scale=args.use_multi_scale, dwt_levels=args.dwt_levels)

    # 设置训练参数
    train_way = args.train_way
    shots = [args.train_shot, args.train_query_shot]

    # 创建数据加载器
    train_loader = dataloaders.meta_train_dataloader(
        data_path=pm.train,
        way=train_way,
        shots=shots,
        transform_type=args.train_transform_type,
        use_dwt=True,
        concat_rgb=True,
        use_multi_scale=args.use_multi_scale,
        dwt_levels=args.dwt_levels
    )

    # 创建模型
    if args.use_dwt_vit:
        if args.use_multi_scale:
            print(f"\n=== 创建多尺度DWT-ViT模型 with 上下文细化 ===")
            print(f"使用模型: {args.vit_model_name}")
            print(f"DWT层数: {args.dwt_levels}")
            print(f"上下文细化: {args.use_context_refinement}")

            model = create_frn_with_multiscale_dwt_vit(
                way=train_way,
                shots=[args.train_shot, args.train_query_shot],
                num_classes=args.num_classes,
                model_name=args.vit_model_name,
                dwt_levels=args.dwt_levels,
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate,
                use_context_refinement=args.use_context_refinement  # 传递参数
            )
        else:
            print(f"\n=== 创建标准DWT-ViT模型 ===")
            print(f"使用模型: {args.vit_model_name}")

            dwt_vit_config = {
                'model_name': args.vit_model_name,
                'pretrained': True,
                'num_classes': args.num_classes,
                'drop_rate': args.drop_rate,
                'attn_drop_rate': args.attn_drop_rate,
                'drop_path_rate': args.drop_path_rate,
                'fusion_layer': 9
            }

            model = FRN(
                way=train_way,
                shots=[args.train_shot, args.train_query_shot],
                use_dwt_vit=True,
                dwt_vit_config=dwt_vit_config,
                use_multi_scale=False
            )

        print("=== 模型创建完成 ===\n")
    else:
        # 使用原始模型
        model = FRN(
            way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet
        )

    # 如果使用稀疏性正则化，修改训练函数
    if args.sparsity_lambda > 0 or args.sparsity_mu > 0:
        print("\n=== 使用带稀疏性正则化的训练 ===")

        def train_with_sparsity(train_loader, model, optimizer, writer, iter_counter):
            """带稀疏性正则化的训练函数"""
            iter_counter, avg_acc, avg_loss = frn_train.default_train(
                train_loader, model, optimizer, writer, iter_counter
            )

            # 添加稀疏性损失
            if model.use_multi_scale:
                sparsity_loss = compute_sparsity_loss(
                    model,
                    lambda_dir=args.sparsity_lambda,
                    mu_scale=args.sparsity_mu
                )

                if sparsity_loss > 0:
                    avg_loss += sparsity_loss.item()
                    writer.add_scalar('sparsity_loss', sparsity_loss.item(), iter_counter)

            return iter_counter, avg_acc, avg_loss

        train_func = partial(train_with_sparsity, train_loader=train_loader)
    else:
        # 使用标准训练函数
        train_func = partial(frn_train.default_train, train_loader=train_loader)

    # 创建训练管理器
    tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)

    # 如果使用warmup，修改优化器
    if args.warmup_epochs > 0 and args.use_dwt_vit:
        print(f"\n=== 设置Warmup学习率调度 ===")
        from torch.optim.lr_scheduler import LambdaLR

        # 获取优化器
        optimizer, base_scheduler = trainer.get_opt(model, args)

        # 创建warmup调度器
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return float(epoch) / float(max(1, args.warmup_epochs))
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # 组合warmup和基础调度器
        class CombinedScheduler:
            def __init__(self, warmup_scheduler, base_scheduler, warmup_epochs):
                self.warmup_scheduler = warmup_scheduler
                self.base_scheduler = base_scheduler
                self.warmup_epochs = warmup_epochs
                self.epoch = 0

            def step(self):
                if self.epoch < self.warmup_epochs:
                    self.warmup_scheduler.step()
                else:
                    self.base_scheduler.step()
                self.epoch += 1

        combined_scheduler = CombinedScheduler(warmup_scheduler, base_scheduler, args.warmup_epochs)

    print("\n=== 开始训练 ===")

    # 训练模型
    tm.train(model)

    # 评估模型
    print("\n开始评估...")
    tm.evaluate(model)

    # 如果是多尺度模型，打印最终的可解释性分析
    if args.use_multi_scale and model.use_multi_scale:
        print("\n=== 最终可解释性分析 ===")

        # 安全加载最佳模型
        model = load_model_safely(model, tm.save_path)
        model.eval()
        model.cuda()

        # 获取一个测试batch
        test_loader = dataloaders.meta_test_dataloader(
            data_path=pm.test,
            way=args.test_way,
            shot=args.test_shot[0],
            pre=args.pre,
            transform_type=args.test_transform_type,
            use_dwt=True,
            concat_rgb=True,
            use_multi_scale=True,
            dwt_levels=args.dwt_levels,
            query_shot=args.test_query_shot,
            trial=1
        )

        for inp, _ in test_loader:
            inp = inp.cuda()
            interp_info = model.get_interpretability_info(inp)

            if interp_info:
                print(f"域权重分析:")
                print(f"  空域权重: {interp_info['domain_weights']['spatial']:.3f}")
                print(f"  频域权重: {interp_info['domain_weights']['frequency']:.3f}")
                print(f"\n各子带贡献度:")
                print(f"  LL: {interp_info['contributions']['LL']:.3f}")
                print(f"  LH: {interp_info['contributions']['LH']:.3f}")
                print(f"  HL: {interp_info['contributions']['HL']:.3f}")
                print(f"  HH: {interp_info['contributions']['HH']:.3f}")
            break


if __name__ == "__main__":
    main()