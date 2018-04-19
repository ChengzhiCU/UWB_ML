from __future__ import print_function
import time
from learning.utils import *
from learning.log import *
from torch.autograd import Variable
import torch
import numpy as np
import config


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
    loss_mse_all = 0
    loss_var_all = 0
    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave, mask, dis = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        mask = Variable(mask.cuda())
        dis = Variable(dis.cuda())

        if 'npn' in args.enc_type:
            a_m, a_s = enc.forward(input)
            #if not args.regression_delta:
            #    a_m = a_m + dis.unsqueeze(1)
            # loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 * mask/ (a_s + 1e-10)
            # + args.lambda_ * torch.log(a_s) * mask)
            loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 * mask / (a_s + 1e-10)
                             + (args.lambda_ * a_s ** 2) * mask)
            loss = loss / torch.sum(mask)

            mse_loss = torch.sum((a_m - labels) ** 2 * mask) / torch.sum(mask)
            var_loss = torch.sum(a_s ** 2 * mask) / torch.sum(mask)
        elif 'combined' in args.enc_type:
            if 'dis' in args.enc_type:
                dis = input[:, 0]
                a_m, a_s = enc.forward((wave, dis))
            else:
                a_m, a_s = enc.forward(wave)

            #if not args.regression_delta:
            #    a_m = a_m + dis.unsqueeze(1)
            loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 * mask / (a_s + 1e-10)
                             + args.lambda_ * torch.log(a_s) * mask)
            # loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 * mask / (a_s + 1e-10)
            #                  + (args.lambda_ * a_s ** 2) * mask)
            loss = loss / torch.sum(mask)

            mse_loss = torch.sum((a_m - labels) ** 2 * mask) / torch.sum(mask)
            var_loss = torch.sum(a_s ** 2 * mask) / torch.sum(mask)
        elif 'cnn' in args.enc_type:
            predict = enc.forward(wave)
            if not args.regression_delta:
                predict = predict + dis.unsqueeze(1)
            loss = full_mse_loss_masked(predict, labels, mask)
        else:
            predict = enc.forward(input)

            if not args.regression_delta:
                predict = predict + dis.unsqueeze(1)
            loss = full_mse_loss_masked(predict, labels, mask)

        for model in models:
            model.zero_grad()
        loss.backward()
        opt_non_discr.step()

        loss_all += loss.data[0]
        if 'npn' in args.enc_type or 'combined' in args.enc_type:
            loss_mse_all += mse_loss.data[0]
            loss_var_all += var_loss.data[0]
        loss_cnt += 1.0
    if 'npn' in args.enc_type or 'combined' in args.enc_type:
        string_out = "{} epoch {}:\ttrain loss = {}\t uncertainty = {} \tmse_square_loss = {}\n" \
              .format(args.enc_type, epoch, loss_all/loss_cnt, round((loss_var_all/loss_cnt) ** 0.5, 3),
                      round(loss_mse_all/loss_cnt, 3))
        print(string_out)
        args.fp.write(string_out)
        return loss_mse_all/loss_cnt
    else:
        string_out = "{} epoch {}:\t train loss = {}\n".format(args.enc_type, epoch, loss_all / loss_cnt)
        print(string_out)
        args.fp.write(string_out)
        return loss_all/loss_cnt


def val_loop(models, data_loader, epoch, args, saveResult=True):
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

    predict_y = []
    variance_y = []
    groundtruth = []
    estimate_d = []

    for idx, icml_data in enumerate(data_loader, 0):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave, _, dis = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        dis = Variable(dis.cuda())

        # predict = enc.forward(input)
        # loss = full_mse_loss(predict, labels)
        if 'npn' in args.enc_type:
            a_m, a_s = enc.forward(input)
            #if not args.regression_delta:
            #    a_m = a_m + input[:, 0].unsqueeze(1)
            loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * torch.log(a_s))
            loss = loss / a_m.size(1) / a_m.size(0)

            mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
            var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)
            abs_loss = torch.sum(torch.abs(a_m - labels)) / a_m.size(1) / a_m.size(0)
        elif 'combined' in args.enc_type:
            if 'dis' in args.enc_type:
                dis_norm = input[:, 0]
                a_m, a_s = enc.forward((wave, dis_norm))
            else:
                a_m, a_s = enc.forward(wave)

            if not args.regression_delta:
                a_m = a_m #+ dis.unsqueeze(1)
                #print(dis[0], a_m[0], labels[0])
            # loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * torch.log(a_s + 1e-10))
            loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * a_s**2)
            loss = loss / a_m.size(1) / a_m.size(0)

            abs_loss = torch.sum(torch.abs(a_m - labels)) / a_m.size(1) / a_m.size(0)
            mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
            var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)
        elif 'cnn' in args.enc_type:
            predict = enc.forward(wave)
            loss = full_mse_loss(predict, labels)
            if not args.regression_delta:
                predict = predict + dis.unsqueeze(1)
            abs_loss = torch.sum(torch.abs(predict - labels)) / labels.size(1) / labels.size(0)
        else:
            predict = enc.forward(input)
            loss = full_mse_loss(predict, labels)
            if not args.regression_delta:
                predict = predict + dis.unsqueeze(1)
            abs_loss = torch.sum(torch.abs(predict - labels)) / labels.size(1) / labels.size(0)

        loss_all += loss.data[0]
        abs_loss_all += abs_loss.data[0]

        if 'npn' in args.enc_type or 'combined' in args.enc_type:
            loss_mse_all += mse_loss.data[0]
            loss_var_all += var_loss.data[0]
        loss_cnt += 1.0

        if saveResult:
            if idx == 0:
                if 'npn' in args.enc_type or 'combined' in args.enc_type:
                    predict_y = a_m.data.cpu().numpy()
                    variance_y = a_s.data.cpu().numpy()
                else:
                    predict_y = predict.data.cpu().numpy()
                groundtruth = labels.data.cpu().numpy()
                estimate_d = dis.data.cpu().numpy()

            else:
                if 'npn' in args.enc_type or 'combined' in args.enc_type:
                    predict_y = np.concatenate((predict_y, a_m.data.cpu().numpy()), axis=0)
                    variance_y = np.concatenate((variance_y, a_s.data.cpu().numpy()), axis=0)
                else:
                    predict_y = np.concatenate((predict_y, predict.data.cpu().numpy()), axis=0)
                groundtruth = np.concatenate((groundtruth, labels.data.cpu().numpy()), axis=0)
                estimate_d = np.concatenate((estimate_d, dis.data.cpu().numpy()), axis=0)

    if saveResult:
        datasave = {}
        datasave['groundtruth'] = groundtruth
        datasave['predict_y'] = predict_y
        datasave['estimate_d'] = estimate_d
        if 'npn' in args.enc_type or 'combined' in args.enc_type:
            datasave['variance_y'] = variance_y
        np.save('../npy_bk/temp_' + args.output.split('/')[-1], datasave)
        import scipy.io
        scipy.io.savemat(os.path.join(config.MAT_PLOT_PATH,
                                      args.parsed_folder.split('/')[-1] + '_' + args.output.split('/')[-1]), datasave)

    if 'npn' in args.enc_type or 'combined' in args.enc_type:
        string_out = "val loss = {}  certainty_variance = {} mse_square_loss = {} meter error = {}\n".format(loss_all/loss_cnt,
                                                                                   round((loss_var_all/loss_cnt) ** 0.5, 3),
                                                                                   round(loss_mse_all/loss_cnt, 3),
                                                                                   round(abs_loss_all / loss_cnt, 3))
        print(string_out)
        args.fp.write(string_out)
        return loss_mse_all/loss_cnt, abs_loss_all / loss_cnt
    else:
        string_out = "val loss = {}  meter_error = {}\n".format(loss_all / loss_cnt, abs_loss_all / loss_cnt)
        print(string_out)
        args.fp.write(string_out)
        return loss_all/loss_cnt, abs_loss_all / loss_cnt
