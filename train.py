from __future__ import print_function
import time, stat, random, shutil, argparse, os
import numpy as np
from learning.datasets_config import get_random_filenames
from learning.utils import *
from learning.datasets import *
from learning.loops import train_loop, val_loop
from learning.vae_loop import vae_train_loop, vae_val_loop
from learning.AE_loop import AE_train_loop, AE_val_loop
from learning.vae_mlp_loop import vaeMlp_train_loop, vaeMlp_val_loop
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
parser.add_argument('--epochs', default=30, type=int, help='number of epochs to run')
parser.add_argument('--seed', default=2333, type=int, help='manual seed')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--cnn_width', default=16, type=int, help='number of channels for first layer cnn')
parser.add_argument('--checkpoint', type=str, help='location of the checkpoint to load')
parser.add_argument('--enc_type', default='combined_dis', type=str, help='type of models') #mlp, cnn, npn, combined_dis
# parser.add_argument('--data_filename', default='all_698.npy', type=str, help='type of models')
parser.add_argument('--data_filename', default='all_436.npy', type=str, help='type of models')
# parser.add_argument('--data_filename', default='all_258.npy', type=str, help='type of models')
parser.add_argument('--loss_type', default='L1', type=str, help='type of models')
parser.add_argument('--output', default=time.strftime('%m-%d-%H-%M'),
                    type=str, help='folder to output model checkpoints')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--use_unlabeled', action='store_true', help='evaluate model on validation set')
parser.add_argument('--val_plot', action='store_true', help='evaluate model on validation set')

parser.add_argument('--train-epoch', default=1, type=int, help='begining epoch No., just for saving model')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--lambda_', default=0.3, type=float, help='ratio of mse and variance')
parser.add_argument('--lambda_vae', default=0.5, type=float, help='ratio of npn and vae loss')
parser.add_argument('--marg_lambda', default=3, type=float, help='ratio of npn and vae loss')
parser.add_argument('--add_noise', default=0, type=float, help='std of noise')
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

if args.data_filename == 'all_698.npy':
    parsed_folder = config.PARSED_FILES_LOSNEW_NLOSOLD
elif args.data_filename == 'all_436.npy':
    parsed_folder = config.PAESED_FILES_6F_NLOS
elif args.data_filename == 'all_258.npy':
    parsed_folder = config.LOS_PAESED_FILES_NEW

args.parsed_folder = parsed_folder
train_dataset = UWBDataset(
    labeled_path=os.path.join(parsed_folder, args.data_filename),
    unlabelled_path=os.path.join(config.UNLABELED_PARSED, 'unlabeled_11.npy'),
    train_index_file=os.path.join(parsed_folder, 'train_tr_ind_sep.npy'),
    regression_delta=args.regression_delta,
    enc_type=args.enc_type,
    used_unlabeled=args.use_unlabeled
    # train_index_file=os.path.join(config.PAESED_FILES, 'train_ind_sep.npy')
)

val_dataset = UWBDataset(
    labeled_path=os.path.join(parsed_folder, args.data_filename),
    unlabelled_path=[],
    train_index_file=os.path.join(parsed_folder, 'train_val_ind_sep.npy'),
    regression_delta=args.regression_delta,
    enc_type=args.enc_type,
    used_unlabeled=False
)

test_dataset = UWBDataset(
    labeled_path=os.path.join(parsed_folder, args.data_filename),
    unlabelled_path=[],
    train_index_file=os.path.join(parsed_folder, 'test_ind_sep.npy'),
    regression_delta=args.regression_delta,
    enc_type=args.enc_type,
    used_unlabeled=False
)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
test_dataloader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

if 'vae' == args.enc_type or 'vae_1' == args.enc_type:
    print('initialize vae')
    enc = nn.DataParallel(VaeEnc(args)).cuda()
    dec = nn.DataParallel(VaeDec(args)).cuda()
    model_names = ['enc', 'dec']
    models = [enc, dec]
    opt_non_D = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
elif 'AE' == args.enc_type:
    print('initialize AE')
    enc = nn.DataParallel(AEEnc(args)).cuda()
    dec = nn.DataParallel(AEDec(args)).cuda()
    model_names = ['enc', 'dec']
    models = [enc, dec]
    opt_non_D = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
elif 'vaemlp' == args.enc_type:
    print('initialize VaeMlp')
    enc = nn.DataParallel(VaeMlpEnc(args)).cuda()
    dec = nn.DataParallel(VaeDec(args)).cuda()   # use the same as vae model is OK
    model_names = ['enc', 'dec']
    models = [enc, dec]
    opt_non_D = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
else:
    enc = nn.DataParallel(Enc(args)).cuda()
    model_names = ['enc']
    models = [enc]
    opt_non_D = optim.Adam(enc.parameters(), lr=args.lr)

optimizers = [opt_non_D]

# lr_scheduler_non_D = lr_scheduler.ExponentialLR(optimizer=opt_non_D, gamma=0.5 ** (1/150))
lr_scheduler_non_D = lr_scheduler.MultiStepLR(optimizer=opt_non_D, milestones=[8, 20], gamma=0.1)
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

best_meter_abs_metric_val = 99999999999
best_meter_abs_metric_test = 99999999999
best_rmse_metric_val = 99999999999
best_rmse_metric_test = 99999999999
best_rmse_epoch = 0
best_epoch = 0

os.makedirs(os.path.join(config.FIG_PATH, args.output.split('/')[-1]), exist_ok=True)
fp = open(os.path.join(args.output, 'log.txt'), 'a')
args.fp = fp
for epoch in range(args.epochs):
    print("")
    if args.enc_type == 'vae' or args.enc_type == 'vae_1':
        train_start_time = time.time()
        train_loss_ave = vae_train_loop(models, train_dataloader, optimizers, lr_schedulers,
                                    epoch, args)
        train_time_cost = time.time() - train_start_time
        infer_start_time = time.time()
        rmse_metric, abs_metric = vae_val_loop(models, val_dataloader, epoch, args)
        infer_time_cost = time.time() - infer_start_time
    elif args.enc_type == 'AE':
        train_start_time = time.time()
        train_loss_ave = AE_train_loop(models, train_dataloader, optimizers, lr_schedulers,
                                    epoch, args)
        train_time_cost = time.time() - train_start_time
        infer_start_time = time.time()
        rmse_metric, abs_metric = AE_val_loop(models, val_dataloader, epoch, args)
        infer_time_cost = time.time() - infer_start_time
    elif args.enc_type == 'vaemlp':
        train_start_time = time.time()
        train_loss_ave = vaeMlp_train_loop(models, train_dataloader, optimizers, lr_schedulers,
                                    epoch, args)
        train_time_cost = time.time() - train_start_time
        infer_start_time = time.time()
        rmse_metric, abs_metric = vaeMlp_val_loop(models, val_dataloader, epoch, args)
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

    if abs_metric < best_meter_abs_metric_val:  # not finished...
        best_meter_abs_metric_val = abs_metric
        best_rmse_metric_val = rmse_metric
        best_epoch = epoch
        save_model(model_names, models, args.output, epoch, best_meter_abs_metric_val)  # save models to one zip file
        if args.enc_type == 'vae' or args.enc_type == 'vae_1':
            best_rmse_metric_test, best_meter_abs_metric_test = vae_val_loop(models, test_dataloader, epoch,
                                                                             args, saveResult=True)
        elif args.enc_type == 'AE':
            best_rmse_metric_test, best_meter_abs_metric_test = AE_val_loop(models, test_dataloader, epoch,
                                                                            args, saveResult=True)
        elif args.enc_type == 'vaemlp':
            best_rmse_metric_test, best_meter_abs_metric_test = vaeMlp_val_loop(models, test_dataloader, epoch,
                                                                                args, saveResult=True)
        else:
            best_rmse_metric_test, best_meter_abs_metric_test = val_loop(models, test_dataloader, epoch,
                                                                         args, saveResult=True)
        str_print = 'test    meter error = {}  rmse loss = {}\n'.format(best_meter_abs_metric_test,
                                                                        best_rmse_metric_test)
        print(str_print)
        args.fp.write(str_print)

total_time_cost = time.time() - start_time
output_str = 'best val rmse loss = {},  epoch = {}\n time cost = {} \n train time = {} \n infer time = {}'\
    .format(best_rmse_metric_val ** 0.5, best_epoch, total_time_cost, train_time_cost, infer_time_cost)
print(output_str)
args.fp.write(output_str)
output_str2 = ' in meter average error = {}\n'.format(best_meter_abs_metric_val)
print(output_str2)
args.fp.write(output_str2)

output_str3 = 'best test meter loss = {}, best rmse loss = {} '.format(best_meter_abs_metric_test, best_rmse_metric_test ** 0.5)
print(output_str3)
args.fp.write(output_str3)
args.fp.close()
print('regress type is delta?', args.regression_delta)
# print arguments
print("Summary of Arguments:")
for key, val in vars(args).items():
    print("{:10} {}".format(key, val))


from visualization.utils import CDF_plot
datasave = np.load('../npy_bk/temp_' + args.output.split('/')[-1] + '.npy')[()]
label = datasave['groundtruth']
predict_y = datasave['predict_y']
CDF_plot(np.abs(predict_y - label), 200, parsed_folder.split('/')[-1] + '_' + args.output.split('/')[-1]
         + str(best_meter_abs_metric_test))


