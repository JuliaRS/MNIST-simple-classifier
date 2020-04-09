import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import logging
import models
from dataset import get_data

def evaluate(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_acc = 0
        val_loss = 0.0
        for x, y in test_loader:
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            _, pred = torch.max(out.data, 1)
            val_acc += (pred == y).numpy().mean()
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
    return val_loss, val_acc

def train(args):
    train_loader, test_loader = get_data()
    writer = SummaryWriter('./logs/{}'.format(datetime.datetime.now().strftime("%d %m %y %H %M")))
    net = getattr(models, args.type)(10)
    writer.add_graph(net, torch.zeros(1, 1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    avg = []
    max_epoch = 10
    for e in range(max_epoch):
        logging.info('run epoch number: {}'.format(e))
        print('run epoch number: {}'.format(e))
        net.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            avg.append(loss.item())
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train/loss', loss.item(), e * len(train_loader) + i)
            writer.add_scalar('Train/lr', scheduler.get_lr()[0], e * len(train_loader) + i)
            if i % 100 == 0:
                logging.debug('Epoch: {}, iteration: {}, loss: {}'.format(e, i, np.mean(avg)))
                val_loss, val_acc = evaluate(net, test_loader, criterion)
                writer.add_scalar('Val/loss', val_loss, e * len(train_loader) + i)
                writer.add_scalar('Val/acc', val_acc, e * len(train_loader) + i)
                logging.debug('Epoch {}, val loss {}, val acc {}'.format(e, val_loss, val_acc))
        scheduler.step()
    logging.debug('FINAL val loss {}, val acc {}'.format(val_loss))
    logging.debug('FINAL val acc {}'.format(val_acc))
