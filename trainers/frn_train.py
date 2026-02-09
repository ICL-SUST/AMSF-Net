import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss, BCEWithLogitsLoss, BCELoss


def default_train(train_loader, model, optimizer, writer, iter_counter):

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()
    lr = optimizer.param_groups[0]['lr']

    # 记录学习率和模型参数
    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)
    writer.add_scalar('alpha', model.r[0].item(), iter_counter)
    writer.add_scalar('beta', model.r[1].item(), iter_counter)

    avg_frn_loss = 0
    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        inp = inp.cuda()

        log_prediction, support = model(inp)

        frn_loss = criterion(log_prediction, target)

        if torch.isnan(frn_loss) or torch.isinf(frn_loss):
            print(f"Warning: Invalid loss at iteration {iter_counter}, skipping...")
            continue

        loss = frn_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_frn_loss += frn_loss.item()
        avg_loss += loss.item()

    num_batches = i + 1
    avg_acc = avg_acc / num_batches
    avg_loss = avg_loss / num_batches
    avg_frn_loss = avg_frn_loss / num_batches

    writer.add_scalar('total_loss', avg_loss, iter_counter)
    writer.add_scalar('frn_loss', avg_frn_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss


def pre_train(train_loader, model, optimizer, writer, iter_counter):

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)
    writer.add_scalar('alpha', model.r[0].item(), iter_counter)
    writer.add_scalar('beta', model.r[1].item(), iter_counter)

    criterion = NLLLoss().cuda()

    avg_loss = 0
    avg_acc = 0

    for i, (inp, target) in enumerate(train_loader):
        iter_counter += 1
        batch_size = target.size(0)
        target = target.cuda()
        inp = inp.cuda()


        log_prediction = model.forward_pretrain(inp)

        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * (torch.sum(torch.eq(max_index, target)).float() / batch_size).item()

        avg_acc += acc
        avg_loss += loss.item()

    num_batches = i + 1
    avg_loss = avg_loss / num_batches
    avg_acc = avg_acc / num_batches

    writer.add_scalar('pretrain_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss


def default_train_with_augmentation(train_loader, model, optimizer, writer, iter_counter):

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)
    writer.add_scalar('alpha', model.r[0].item(), iter_counter)
    writer.add_scalar('beta', model.r[1].item(), iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        inp = inp.cuda()

        log_prediction, support = model(inp)
        loss = criterion(log_prediction, target)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss.item()

    num_batches = i + 1
    avg_acc = avg_acc / num_batches
    avg_loss = avg_loss / num_batches

    writer.add_scalar('train_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss


class LabelSmoothingNLLLoss(nn.Module):


    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):

        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * self.confidence + self.smoothing / n_class
        return -(one_hot * pred).sum(dim=1).mean()


def default_train_with_label_smoothing(train_loader, model, optimizer, writer, iter_counter, smoothing=0.15):

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = LabelSmoothingNLLLoss(smoothing=smoothing).cuda()
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr', lr, iter_counter)
    writer.add_scalar('scale', model.scale.item(), iter_counter)
    writer.add_scalar('alpha', model.r[0].item(), iter_counter)
    writer.add_scalar('beta', model.r[1].item(), iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        inp = inp.cuda()

        log_prediction, support = model(inp)
        loss = criterion(log_prediction, target)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss.item()

    num_batches = i + 1
    avg_acc = avg_acc / num_batches
    avg_loss = avg_loss / num_batches

    writer.add_scalar('train_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss