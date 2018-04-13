from __future__ import print_function
import time
from learning.utils import *
from learning.log import *
from torch.autograd import Variable
import torch


def train_loop_combined(models, data_loader, optimizers, lr_schedulers, epoch, args):
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
    loss_mse_all = 0
    loss_var_all = 0
    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())

        if 'dis' in args.enc_type:
            dis = input[:, 0]
            a_m, a_s = enc.forward((wave, dis))
        else:
            a_m, a_s = enc.forward(wave)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)
        loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-8) + args.lambda_ * torch.log(a_s + 1e-8))
        loss = loss / a_m.size(1) / a_m.size(0)

        mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
        var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)

        for model in models:
            model.zero_grad()
        loss.backward()
        opt_non_discr.step()

        loss_all += loss.data[0]

        loss_mse_all += mse_loss.data[0]
        loss_var_all += var_loss.data[0]
        loss_cnt += 1.0

    string_out = "{} epoch {}:                train loss = {} certainty = {}  mse_square_loss = {}\n" \
          .format(args.enc_type, epoch, loss_all/loss_cnt, (loss_var_all/loss_cnt) ** 0.5, loss_mse_all/loss_cnt)
    print(string_out)
    args.fp.write(string_out)
    return loss_mse_all/loss_cnt


def val_loop_combined(models, data_loader, epoch, args):
    for model in models:
        model.eval() ####depends
        set_dropout_mode(model, False)

    enc = models[0]
    num_per_epoch = len(data_loader)
    loss_all = 0
    abs_loss_all = 0
    loss_cnt = 0
    loss_mse_all = 0
    loss_var_all = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())

        if 'dis' in args.enc_type:
            dis = input[:, 0]
            a_m, a_s = enc.forward((wave, dis))
        else:
            a_m, a_s = enc.forward(wave)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)
        # loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * torch.log(a_s + 1e-10))
        loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * a_s**2)
        loss = loss / a_m.size(1) / a_m.size(0)

        abs_loss = torch.sum(torch.abs(a_m - labels)) / a_m.size(1) / a_m.size(0)
        mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
        var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)

        loss_all += loss.data[0]
        abs_loss_all += abs_loss.data[0]

        loss_mse_all += mse_loss.data[0]
        loss_var_all += var_loss.data[0]
        loss_cnt += 1.0

    string_out = "val loss = {}  certainty_variance = {} mse_square_loss = {} meter error = {}\n".format(loss_all/loss_cnt,
                                                                               (loss_var_all/loss_cnt) ** 0.5,
                                                                               loss_mse_all/loss_cnt,
                                                                               abs_loss_all / loss_cnt)
    print(string_out)
    args.fp.write(string_out)
    return loss_mse_all/loss_cnt, abs_loss_all / loss_cnt

