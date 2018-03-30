from __future__ import print_function
import time
from learning.utils import *
from learning.log import *
from torch.autograd import Variable
import torch


def vae_train_loop(models, data_loader, optimizers, lr_schedulers, epoch, args):
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
    loss_ELBO_all = 0
    loss_marginal_likelihood_all = 0
    loss_KL_divergence_all = 0

    loss_cnt = 0

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        dis = input[:, 0]

        if args.add_noise > 0:
            wave_in = wave + Variable(torch.normal(means=torch.zeros(wave.size()), std=args.add_noise).cuda())
        else:
            wave_in = wave

        a_m, a_s, mu, sigma, z = enc.forward((wave_in, dis))
        y = dec.forward(z)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)

        marginal_likelihood = torch.sum((y - wave) ** 2) / wave.size(0) / wave.size(1)
        KL_divergence = - torch.mean(0.5 * torch.sum(1 + torch.log(1e-8 + sigma ** 2) - mu ** 2 - sigma ** 2))

        ELBO = marginal_likelihood + KL_divergence

        npn_loss = torch.sum((1 - args.lambda_) * (a_m - labels) ** 2 / (a_s + 1e-10) + args.lambda_ * torch.log(a_s))
        npn_loss = npn_loss / a_m.size(1) / a_m.size(0)

        mse_loss = torch.sum((a_m - labels) ** 2) / a_m.size(1) / a_m.size(0)
        abs_loss = torch.sum(torch.abs(a_m - labels)) / a_m.size(1) / a_m.size(0)
        var_loss = torch.sum(a_s ** 2) / a_m.size(1) / a_m.size(0)

        loss = (1 - args.lambda_vae) * npn_loss + args.lambda_vae * ELBO

        for model in models:
            model.zero_grad()
        loss.backward()
        opt_non_discr.step()

        loss_all += loss.data[0]
        loss_mse_all += mse_loss.data[0]
        loss_abs_all += abs_loss.data[0]
        loss_var_all += var_loss.data[0]
        loss_ELBO_all += ELBO.data[0]
        loss_marginal_likelihood_all += marginal_likelihood.data[0]
        loss_KL_divergence_all += KL_divergence.data[0]

        loss_cnt += 1.0

        string_out = "{} epoch {}:                train loss = {} certainty = {}  mse_square_loss = {} average_meter_loss = {}\n" \
                     "ELBO = {}   marginal_likelihood = {}   KL_divergence = {}\n" \
            .format(args.enc_type, epoch, loss_all / loss_cnt, (loss_var_all / loss_cnt) ** 0.5,
                    loss_mse_all / loss_cnt, loss_abs_all / loss_cnt, loss_ELBO_all / loss_cnt, loss_marginal_likelihood_all / loss_cnt,
                    loss_KL_divergence_all / loss_cnt)
        print(string_out)
        args.fp.write(string_out)
        return loss_mse_all / loss_cnt


def vae_val_loop(models, data_loader, epoch, args):
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

    for idx, icml_data in enumerate(data_loader, 1):
        if idx > num_per_epoch:
            break
        input, labels, subject, wave = icml_data
        input = Variable(input.cuda())
        labels = Variable(labels.cuda())
        wave = Variable(wave.cuda())
        dis = input[:, 0]

        a_m, a_s, mu, sigma, z = enc.forward((wave, dis))
        y = dec.forward(z)

        if not args.regression_delta:
            a_m = a_m + input[:, 0].unsqueeze(1)

        marginal_likelihood = torch.sum((y - wave) ** 2) / wave.size(0) / wave.size(1)
        KL_divergence = - torch.mean(0.5 * torch.sum(1 + torch.log(1e-8 + sigma ** 2) - mu ** 2 - sigma ** 2))

        ELBO = marginal_likelihood + KL_divergence

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
        loss_ELBO_all += ELBO.data[0]
        loss_marginal_likelihood_all += marginal_likelihood.data[0]
        loss_KL_divergence_all += KL_divergence.data[0]

        loss_cnt += 1.0

        string_out = "val loss = {} certainty = {}  mse_square_loss = {}  average meter = {}\n" \
                     "ELBO = {}   marginal_likelihood = {}   KL_divergence = {}\n" \
            .format(loss_all / loss_cnt, (loss_var_all / loss_cnt) ** 0.5,
                    loss_mse_all / loss_cnt, loss_abs_all / loss_cnt, loss_ELBO_all / loss_cnt, loss_marginal_likelihood_all / loss_cnt,
                    loss_KL_divergence_all / loss_cnt)
        print(string_out)
        args.fp.write(string_out)
        return loss_mse_all / loss_cnt, loss_abs_all / loss_cnt


