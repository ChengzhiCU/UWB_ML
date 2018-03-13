from __future__ import print_function
import time, stat, random, shutil, argparse, os
import numpy as np
from learning.datasets_config import get_random_filenames

import config
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data

parser = argparse.ArgumentParser(description='RF-Sleep Training Script')
parser.add_argument('--workers', '-j', default=2, type=int, help='number of data loading workers')
parser.add_argument('--batch', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', default=2000, type=int, help='number of epochs to run')
parser.add_argument('--seed', default=2000, type=int, help='manual seed')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--checkpoint', type=str, help='location of the checkpoint to load')
parser.add_argument('--output', default=time.strftime('%m-%d-%H-%M'),
                    type=str, help='folder to output model checkpoints')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--train-epoch', default=1, type=int, help='begining epoch No., just for saving model')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')

parser.set_defaults(augment=True)
args = parser.parse_args()


np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(int(args.seed))

# setup output folder
args.output = os.path.join(MODEL_PATH, args.output)
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

train_filenames, val_filenames = get_random_filenames(args)


