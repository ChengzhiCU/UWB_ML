from __future__ import print_function
import time
from learning.utils import *
from learning.log import *
from torch.autograd import Variable

def train_loop(models, data_loader, optimizers, lr_schedulers, epoch, args):
    for model in models:
        model.train()
        set_dropout_mode(model, True)

    enc = models[0]
    opt_non_discr = optimizers[0]
    lr_scheduler_non = lr_schedulers[0]

    # schedule learning rate
    lr_scheduler_non.step()

    num_per_epoch = len(data_loader)
    loss_all = 0
    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())

        predict = enc.forward(input)
        loss = full_mse_loss(predict, labels)

        for model in models:
            model.zero_grad()
        loss.backward()
        opt_non_discr.step()

        loss_all += loss.data[0]
        loss_cnt += 1.0

    print("epoch {}:                train loss = {}".format(epoch, loss_all/loss_cnt))
    return loss_all/loss_cnt


def val_loop(models, data_loader, epoch, args):
    for model in models:
        model.eval() ####depends
        set_dropout_mode(model, False)

    enc = models[0]
    num_per_epoch = len(data_loader)
    loss_all = 0
    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())

        predict = enc.forward(input)
        loss = full_mse_loss(predict, labels)

        for model in models:
            model.zero_grad()

        loss_all += loss.data[0]
        loss_cnt += 1.0

    print("epoch {}: val loss = {}".format(epoch, loss_all/loss_cnt))
    return loss_all/loss_cnt
