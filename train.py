from __future__ import print_function
import time, stat, random, shutil, argparse, os
import numpy as np
from learning.datasets_config import get_random_filenames
from learning.utils import *
from learning.datasets import *
from learning.loops import train_loop, val_loop
from learning.vae_loop import vae_train_loop, vae_val_loop
from learning.models import *
import time

import config
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
import torch

parser = argparse.ArgumentParser(description='RF-Sleep Training Script')
parser.add_argument('--workers', '-j', default=1, type=int, help='number of data loading workers')
parser.add_argument('--batch', type=int, default=64, help='input batch size')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to run')
parser.add_argument('--seed', default=2000, type=int, help='manual seed')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--cnn_width', default=16, type=int, help='number of channels for first layer cnn')
parser.add_argument('--checkpoint', type=str, help='location of the checkpoint to load')
parser.add_argument('--enc_type', default='mlp', type=str, help='type of models')
parser.add_argument('--output', default=time.strftime('%m-%d-%H-%M'),
                    type=str, help='folder to output model checkpoints')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--train-epoch', default=1, type=int, help='begining epoch No., just for saving model')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--lambda_', default=0.3, type=float, help='ratio of mse and variance')
parser.add_argument('--lambda_vae', default=0.5, type=float, help='ratio of npn and vae loss')
parser.add_argument('--add_noise', default=0.2, type=float, help='std of noise')
parser.add_argument('--regression_delta', default=False, type=bool, help='Regress error or not')

parser.set_defaults(augment=True)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(int(args.seed))

# setup output folder
args.output = os.path.join(MODEL_PATH, args.output + '_' + args.enc_type)
if os.path.exists(args.output):
    if query_yes_no('overwrite previous folder?'):
        shutil.rmtree(args.output)
        if os.path.exists(args.output + '_val'):
            shutil.rmtree(args.output + '_val')
        print(args.output + ' removed.\n')
    else:
        raise RuntimeError('Output folder {} already exists'.format(args.output))

os.makedirs(args.output, mode=0o770)
os.makedirs(args.output + '_val', mode=0o770)

# copy src files
if args.checkpoint is None:
    shutil.copytree('.', os.path.join(args.output, 'src'))
    os.chmod(os.path.join(args.output, 'src'), stat.S_IRWXU)  # chmod 700 src_folder

# print arguments
print("Summary of Arguments:")
for key, val in vars(args).items():
    print("{:10} {}".format(key, val))

start_time = time.time()
# train_filenames, val_filenames = get_random_filenames(args)
train_dataset = UWBDataset(
    labeled_path=os.path.join(config.PAESED_FILES, 'all_336.npy'),
    unlabelled_path=[],
    train_index_file=os.path.join(config.PAESED_FILES, 'train_ind_sep.npy'),
    regression_delta=args.regression_delta
    # train_index_file=os.path.join(config.PAESED_FILES, 'train_ind_sep.npy')
)

val_dataset = UWBDataset(
    labeled_path=os.path.join(config.PAESED_FILES, 'all_336.npy'),
    unlabelled_path=[],
    train_index_file=os.path.join(config.PAESED_FILES, 'test_ind_sep.npy'),
    regression_delta=args.regression_delta
    # train_index_file=os.path.join(config.PAESED_FILES, 'test_ind_sep.npy')
)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)
val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)
if 'vae' in args.enc_type:
    print('initialize vae')
    enc = nn.DataParallel(VaeEnc(args)).cuda()
    dec = nn.DataParallel(VaeDec(args)).cuda()
    model_names = ['enc', 'dec']
    models = [enc, dec]
    opt_non_D = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
else:
    enc = nn.DataParallel(Enc(args)).cuda()
    model_names = ['enc']
    models = [enc]
    opt_non_D = optim.Adam(enc.parameters(), lr=args.lr)

optimizers = [opt_non_D]

lr_scheduler_non_D = lr_scheduler.ExponentialLR(optimizer=opt_non_D, gamma=0.5 ** (1/100))
lr_schedulers = [lr_scheduler_non_D]

# optionally load model from a checkpoint
if args.checkpoint:
    if os.path.isfile(args.checkpoint):
        load_model(args.checkpoint, models, model_names)
    else:
        raise(RuntimeError("no checkpoint found at '{}'".format(args.checkpoint)))

# evaluation model
if args.evaluate:
    if not args.checkpoint:
        raise RuntimeError(RuntimeWarning("no loaded model"))
    validation_log, _ = val_loop(models, val_dataloader, 1, args)
    with open('{}/log_validation.txt'.format(args.output), 'w') as f:
        f.write('validation_log:\n{}'.format(validation_log))
    exit(0)

best_meter_abs_metric = 99999999999
best_rmse_metric = 99999999999
best_rmse_epoch = 0
best_epoch = 0

fp = open(os.path.join(args.output, 'log.txt'), 'a')
args.fp = fp
for epoch in range(args.epochs):
    print("")
    if args.enc_type == 'vae':
        train_start_time = time.time()
        train_loss_ave = vae_train_loop(models, train_dataloader, optimizers, lr_schedulers,
                                    epoch, args)
        train_time_cost = time.time() - train_start_time
        infer_start_time = time.time()
        rmse_metric, abs_metric = vae_val_loop(models, val_dataloader, epoch, args)
        infer_time_cost = time.time() - infer_start_time
    else:
        train_start_time = time.time()
        train_loss_ave = train_loop(models, train_dataloader, optimizers, lr_schedulers,
                                            epoch, args)
        train_time_cost = time.time() - train_start_time
        infer_start_time = time.time()
        rmse_metric, abs_metric = val_loop(models, val_dataloader, epoch, args)
        infer_time_cost = time.time() - infer_start_time
    print('train time = {}   infer time = {}'.format(train_time_cost, infer_time_cost))

    if abs_metric < best_meter_abs_metric:
        best_meter_abs_metric = abs_metric
        best_epoch = epoch
        save_model(model_names, models, args.output, epoch, best_meter_abs_metric)  # save models to one zip file

    if rmse_metric < best_rmse_metric:
        best_rmse_metric = rmse_metric
        best_rmse_epoch = epoch

total_time_cost = time.time() - start_time
output_str = 'best test rmse loss = {},  epoch = {}\n time cost = {} \n train time = {} \n infer time = {}'\
    .format(best_rmse_metric ** 0.5, best_rmse_epoch, total_time_cost, train_time_cost, infer_time_cost)
print(output_str)
args.fp.write(output_str)
output_str2 = ' in meter average error = {} epoch = {}\n'.format(best_meter_abs_metric, best_epoch)
print(output_str2)
args.fp.write(output_str2)
args.fp.close()
print('regress type is delta?', args.regression_delta)

