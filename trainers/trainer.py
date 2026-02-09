import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .eval import meta_test

sys.path.append('..')
from datasets import dataloaders


def get_logger(filename):
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam', 'sgd'], default='adam')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=5e-5)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.5)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=8000)
    parser.add_argument("--stage", help="number lr stages", type=int,default=3)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-5)
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=500)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", action="store_true", default=False)
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int,default=32)
    parser.add_argument('--decay_epoch', nargs='+', help='epochs that cut lr', type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep",
                        action="store_true")
    parser.add_argument("--train_way", help="training way", type=int, default=4)
    parser.add_argument("--test_way", help="test way", type=int, default=4)
    parser.add_argument("--train_shot",
                        help="number of support images per class for meta-training and meta-testing during validation",
                        type=int,default=5)
    parser.add_argument("--test_shot", nargs='+',
                        help="number of support images per class for meta-testing during final test", type=int, default=[1,5])
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int,
                        default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int,
                        default=15)
    parser.add_argument("--train_transform_type", help="size transformation type during training", type=int,default=2)
    parser.add_argument("--test_transform_type", help="size transformation type during inference", type=int,default=2)
    parser.add_argument("--val_trial", help="number of meta-testing episodes during validation", type=int, default=300)
    parser.add_argument("--detailed_name", help="whether include training details in the name", action="store_true")

    # DWT-ViT相关参数
    parser.add_argument("--use_dwt_vit", help="use DWT-ViT as feature extractor", action="store_true",default=True)
    parser.add_argument("--dwt_vit_pretrained", help="path to pretrained DWT-ViT weights", type=str, default=r'F:\ViT-FRN\vit_base_patch16_224_in21k.pth')
    parser.add_argument("--num_classes", help="number of classes for DWT-ViT (e.g., 4 for MRI)", type=int, default=4)
    parser.add_argument("--use_dwt_transform", help="use DWT transform in data preprocessing", action="store_true",default=True)

    args = parser.parse_args()

    return args


def get_opt(model, args):
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.decay_epoch is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.gamma)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch, gamma=args.gamma)

    return optimizer, scheduler


class Path_Manager:
    def __init__(self, fewshot_path, args):
        self.train = os.path.join(fewshot_path, 'train')

        if args.pre:
            self.test = os.path.join(fewshot_path, 'test_pre')
            self.val = os.path.join(fewshot_path, 'val_pre') if not args.no_val else self.test
        else:
            self.test = os.path.join(fewshot_path, 'test')
            self.val = os.path.join(fewshot_path, 'val') if not args.no_val else self.test


class Train_Manager:
    def __init__(self, args, path_manager, train_func):
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.set_device(args.gpu)

        # 根据使用的特征提取器设置名称
        if args.use_dwt_vit:
            name = 'DWT-ViT'
        elif args.resnet:
            name = 'ResNet-12'
        else:
            name = 'Conv-4'

        if args.detailed_name:
            if args.decay_epoch is not None:
                temp = ''
                for i in args.decay_epoch:
                    temp += ('_' + str(i))

                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-drop%s-decay_%.0e-way_%d' % (args.opt,
                                                                                      args.lr, args.gamma, args.epoch,
                                                                                      temp, args.weight_decay,
                                                                                      args.train_way)
            else:
                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-stage_%d-decay_%.0e-way_%d' % (args.opt,
                                                                                        args.lr, args.gamma, args.epoch,
                                                                                        args.stage, args.weight_decay,
                                                                                        args.train_way)

            name = "%s-%s" % (name, suffix)

        self.logger = get_logger('%s.log' % (name))
        self.save_path = 'model_%s.pth' % (name)
        self.writer = SummaryWriter('log_%s' % (name))

        self.logger.info('display all the hyper-parameters in args:')
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg), str(value)))
        self.logger.info('------------------------')
        self.args = args
        self.train_func = train_func
        self.pm = path_manager

    def train(self, model):
        args = self.args
        train_func = self.train_func
        writer = self.writer
        save_path = self.save_path
        logger = self.logger

        optimizer, scheduler = get_opt(model, args)

        val_shot = args.train_shot
        test_way = args.test_way

        best_val_acc = 0
        best_epoch = 0

        model.train()
        model.cuda()

        iter_counter = 0

        if args.decay_epoch is not None:
            total_epoch = args.epoch
        else:
            total_epoch = args.epoch * args.stage

        logger.info("start training!")


        from tqdm import tqdm
        epoch_bar = tqdm(range(total_epoch), desc="Training", ncols=150)

        for e in epoch_bar:
            iter_counter, train_acc, train_loss = train_func(model=model,
                                                             optimizer=optimizer,
                                                             writer=writer,
                                                             iter_counter=iter_counter)


            epoch_bar.set_postfix({
                'acc': f'{train_acc:.2f}%',
                'loss': f'{train_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })


            if (e + 1) % 100 == 0:
                logger.info(f"epoch {e + 1}/{total_epoch}, loss: {train_loss:.4f}, acc: {train_acc:.2f}%")

            if (e + 1) % args.val_epoch == 0:
                logger.info("")
                logger.info("epoch %d/%d, iter %d:" % (e + 1, total_epoch, iter_counter))
                logger.info("train_acc: %.3f, train_loss: %.4f" % (train_acc, train_loss))

                model.eval()
                with torch.no_grad():

                    val_acc, val_interval = meta_test(
                        data_path=self.pm.val,
                        model=model,
                        way=test_way,
                        shot=val_shot,
                        pre=args.pre,
                        transform_type=args.test_transform_type,
                        use_dwt=args.use_dwt_vit,
                        use_multi_scale=args.use_multi_scale if hasattr(args, 'use_multi_scale') else False,
                        dwt_levels=args.dwt_levels if hasattr(args, 'dwt_levels') else 3,
                        query_shot=args.test_query_shot,
                        trial=args.val_trial
                    )
                    writer.add_scalar('val_%d-way-%d-shot_acc' % (test_way, val_shot),
                                      val_acc, iter_counter)
                    logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f' %
                                (test_way, val_shot, val_acc, val_interval))


                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = e + 1
                        if not args.no_val:
                            torch.save(model.state_dict(), save_path)
                        logger.info('BEST!')

                model.train()

            scheduler.step()

        logger.info('training finished!')
        if args.no_val:
            torch.save(model.state_dict(), save_path)

        logger.info('------------------------')
        logger.info(('the best epoch is %d/%d') % (best_epoch, total_epoch))
        logger.info(('the best %d-way %d-shot val acc is %.3f') %
                    (test_way, val_shot, best_val_acc))

    def evaluate(self, model):
        logger = self.logger
        args = self.args

        logger.info('------------------------')
        logger.info('evaluating on test set:')

        with torch.no_grad():

            checkpoint = torch.load(self.save_path, weights_only=False)
            model_dict = model.state_dict()


            filtered_dict = {}
            skipped_keys = []

            for key, value in checkpoint.items():
                if 'cached_' in key:
                    logger.info(f"跳过缓存buffer: {key}")
                    skipped_keys.append(key)
                    continue

                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        filtered_dict[key] = value
                    else:
                        logger.info(f"警告: 跳过形状不匹配的参数: {key}")
                        skipped_keys.append(key)
                else:
                    if not key.startswith('cached_'):
                        logger.info(f"警告: checkpoint中的键 '{key}' 在模型中不存在")
                    skipped_keys.append(key)

            model_dict.update(filtered_dict)

            model.load_state_dict(model_dict, strict=False)

            if skipped_keys:
                logger.info(f"总共跳过了 {len(skipped_keys)} 个键")

            model.eval()
            model.cuda()

            for shot in args.test_shot:
                mean, interval = meta_test(
                    data_path=self.pm.test,
                    model=model,
                    way=args.test_way,
                    shot=shot,
                    pre=args.pre,
                    transform_type=args.test_transform_type,
                    use_dwt=args.use_dwt_vit,
                    use_multi_scale=args.use_multi_scale if hasattr(args, 'use_multi_scale') else False,
                    dwt_levels=args.dwt_levels if hasattr(args, 'dwt_levels') else 3,
                    query_shot=args.test_query_shot,
                    trial=10000
                )

                logger.info('%d-way-%d-shot acc: %.2f\t%.2f' % (args.test_way, shot, mean, interval))