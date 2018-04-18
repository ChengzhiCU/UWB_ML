from __future__ import print_function
import time
from learning.utils import *
from learning.log import *
from torch.autograd import Variable
import torch
import numpy as np
import config

def AE_train_loop(models, data_loader, optimizers, lr_schedulers, epoch, args):
    for model in models:
        model.train()
        set_dropout_mode(model, True)

    enc, dec = models
    opt_non_discr = optimizers[0]
    lr_scheduler_non = lr_schedulers[0]

    # schedule learning rate
    lr_scheduler_non.step()

    num_per_epoch = len(data_loader)
    loss_all = 0
    loss_mse_all = 0
    loss_abs_all = 0
    loss_var_all = 0
    loss_marginal_likelihood_all = 0

    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave, mask, _ = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        mask = Variable(mask.cuda())
        dis = input[:, 0]

        if args.add_noise > 0:
            wave_in = wave + Variable(torch.normal(means=torch.zeros(wave.size()), std=args.add_noise).cuda())
        else:
            wave_in = wave

        a_m, a_s, z = enc.forward((wave_in, dis))
        y = dec.forward(z)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)

        marginal_likelihood = torch.sum((y - wave) ** 2) / wave.size(0) / wave.size(1)

        npn_loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 * mask / (a_s + 1e-10)
                             + args.lambda_ * torch.log(a_s) * mask)
        npn_loss = npn_loss / torch.sum(mask)

        mse_loss = torch.sum((a_m - labels) ** 2 * mask) / torch.sum(mask)
        abs_loss = torch.sum(torch.abs(a_m - labels) * mask) / torch.sum(mask)
        var_loss = torch.sum(a_s ** 2 * mask) / torch.sum(mask)

        loss = (1 - args.lambda_vae) * npn_loss + args.lambda_vae * marginal_likelihood

        for model in models:
            model.zero_grad()
        loss.backward()
        opt_non_discr.step()

        loss_all += loss.data[0]
        loss_mse_all += mse_loss.data[0]
        loss_abs_all += abs_loss.data[0]
        loss_var_all += var_loss.data[0]
        loss_marginal_likelihood_all += marginal_likelihood.data[0]

        loss_cnt += 1.0

    string_out = "{} epoch {}:\ttrain loss = {} \t certainty = {} \t  mse_square_loss = {} \t average_meter_loss = {} \
                 marginal_likelihood = {}\n" \
        .format(args.enc_type, epoch, round(loss_all / loss_cnt, 3), round((loss_var_all / loss_cnt) ** 0.5),
                round(loss_mse_all / loss_cnt, 3), round(loss_abs_all / loss_cnt, 4),
                round(loss_marginal_likelihood_all / loss_cnt, 4))
    print(string_out)
    args.fp.write(string_out)
    return loss_mse_all / loss_cnt


def AE_val_loop(models, data_loader, epoch, args, saveResult=True):
    for model in models:
        model.train()
        set_dropout_mode(model, True)

    enc, dec = models

    num_per_epoch = len(data_loader)
    loss_all = 0
    loss_mse_all = 0
    loss_abs_all = 0
    loss_var_all = 0
    loss_ELBO_all = 0
    loss_marginal_likelihood_all = 0
    loss_KL_divergence_all = 0

    loss_cnt = 0
    predict_y = []
    groundtruth = []
    pred_wave = []
    raw_wave = []
    variance_y = []

    for idx, icml_data in enumerate(data_loader, 0):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave, _, _ = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        dis = input[:, 0]

        a_m, a_s, z = enc.forward((wave, dis))
        y = dec.forward(z)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)

        marginal_likelihood = torch.sum((y - wave) ** 2) / wave.size(0) / wave.size(1)

        ELBO = marginal_likelihood

        npn_loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * torch.log(a_s))
        npn_loss = npn_loss / a_m.size(1) / a_m.size(0)

        mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
        abs_loss = torch.sum(torch.abs(a_m - labels)) / a_m.size(1) / a_m.size(0)
        var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)

        loss = (1 - args.lambda_vae) * npn_loss + args.lambda_vae * ELBO

        loss_all += loss.data[0]
        loss_mse_all += mse_loss.data[0]
        loss_abs_all += abs_loss.data[0]
        loss_var_all += var_loss.data[0]
        loss_marginal_likelihood_all += marginal_likelihood.data[0]

        loss_cnt += 1.0

        if saveResult:
            if idx == 0:
                predict_y = a_m.data[0]
                variance_y = a_s.data[0]
                groundtruth = labels.data[0]
                raw_wave = np.expand_dims(wave.data[0], axis=0)
                pred_wave = np.expand_dims(y.data[0], axis=0)
            else:
                predict_y = np.concatenate((predict_y, a_m.data[0]), axis=0)
                variance_y = np.concatenate((variance_y, a_s.data[0]), axis=0)
                groundtruth = np.concatenate((groundtruth, labels.data[0]), axis=0)
                if idx % 20 == 0:
                    if args.val_plot:
                        import matplotlib.pyplot as plt
                        plt.subplot(2,1,1)
                        print('plotshape', wave.data[0].cpu().numpy().shape)
                        plt.plot(wave.data[0].cpu().numpy())
                        plt.title('raw')
                        plt.subplot(2, 1, 2)
                        plt.plot(y.data[0].cpu().numpy())
                        plt.title('pred')
                        plt.savefig(os.path.join(config.FIG_PATH, args.output.split('/')[-1],
                                                 str(epoch) + '_' + str(idx)))
                        plt.gcf().clear()
                    raw_wave = np.concatenate((raw_wave, np.expand_dims(wave.data[0], axis=0)), axis=0)
                    pred_wave = np.concatenate((pred_wave, np.expand_dims(y.data[0], axis=0)), axis=0)


    if saveResult:
        datasave = {}
        datasave['groundtruth'] = groundtruth
        datasave['predict_y'] = predict_y
        datasave['variance_y'] = variance_y
        datasave['raw_wave'] = raw_wave
        datasave['pred_wave'] = pred_wave

        np.save('../npy_bk/temp_' + args.output.split('/')[-1], datasave)
        import scipy.io
        scipy.io.savemat(os.path.join(config.MAT_PLOT_PATH,
                                      args.parsed_folder.split('/')[-1] + '_' + args.output.split('/')[-1]), datasave)

    string_out = "val loss = {} certainty = {}  mse_square_loss = {}  average meter = {}\n" \
                 "ELBO = {}   marginal_likelihood = {}   KL_divergence = {}\n" \
        .format(loss_all / loss_cnt, (loss_var_all / loss_cnt) ** 0.5,
                loss_mse_all / loss_cnt, loss_abs_all / loss_cnt, loss_ELBO_all / loss_cnt, loss_marginal_likelihood_all / loss_cnt,
                loss_KL_divergence_all / loss_cnt)
    print(string_out)
    args.fp.write(string_out)
    return loss_mse_all / loss_cnt, loss_abs_all / loss_cnt


