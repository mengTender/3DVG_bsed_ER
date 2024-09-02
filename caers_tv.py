import argparse
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from datasets import MyCAERDataset
from models.model import ModelForER, collate_fn
from loss_funcs import MultiplicativeLoss
from caer_ext.parse_config import ConfigParser
from caer_ext.trainer import Trainer
from tqdm import tqdm

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def train(config):
    model = ModelForER(out_feature=7)
    logger = config.get_logger('train')
    train_set = MyCAERDataset('./dataset/caer-s', train='train')
    val_set = MyCAERDataset(root='./dataset/caer-s', train='val')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=collate_fn)
    logger.info(model)
    cross_entropy_loss = nn.CrossEntropyLoss()
    multip_loss = MultiplicativeLoss(num_modalities=2)
    metrics = ['accuracy']
    optimizer = optim.Adam(params=model, weight_decay=0.05, lr=1e-3)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, [cross_entropy_loss, multip_loss], metrics, optimizer,
                      config=config,
                      data_loader=train_loader,
                      lr_scheduler=lr_scheduler,
                      valid_data_loader=val_loader)
    trainer.train()


def test(config, bs):
    logger = config.get_logger('test')

    # setup data_loader instances
    test_set = MyCAERDataset(root='./dataset/caer-s', train='test')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # build model architecture
    model = ModelForER(out_feature=7)

    # get function handles of loss and metrics
    cross_entropy_loss = nn.CrossEntropyLoss()
    multip_loss = MultiplicativeLoss(num_modalities=2)
    metric_fns = [accuracy,]

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader)):
            for key in data.keys():
                data[key] = data[key].to(device)
            labels = labels.to(device)
            peb_out, ceb_out, output = model(data)
            # computing loss, metrics on test set
            loss = cross_entropy_loss(output, labels)+multip_loss(peb_out, ceb_out)
            total_loss += loss.item() * bs
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, labels) * bs

    n_samples = len(test_loader)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)




