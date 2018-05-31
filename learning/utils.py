from config import *
import torch
import torch.nn.functional as F
import zipfile, tempfile, glob
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def save_model(model_names, models, output_folder, epoch, metric):
    model_path = '{}/best_model_epoch{}_{:.4f}.zip'.format(output_folder, epoch, metric)
    for model, model_name in zip(models, model_names):
        with open('{}/model_epoch_{}.pth'.format(output_folder, model_name), 'wb') as f:
            torch.save(model.state_dict(), f)
    for filename in glob.glob(os.path.join(output_folder, 'best_model*.zip')):
        os.remove(filename)
    with zipfile.ZipFile(model_path, 'w') as f:
        for model_name in model_names:
            f.write('{}/model_epoch_{}.pth'.format(output_folder, model_name), model_name)
            os.remove('{}/model_epoch_{}.pth'.format(output_folder, model_name))


def load_model(model_file, models, model_names):
    with zipfile.ZipFile(model_file) as f1:
        for model, model_name in zip(models, model_names):
            tempname = tempfile.mktemp(model_name)
            with open(tempname, 'wb') as f2:
                f2.write(f1.read(model_name))
            model.load_state_dict(torch.load(tempname))
            os.remove(tempname)


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def masked_cross_entropy(pred, label, mask):
    """ get masked cross entropy loss, for those training data padded with zeros at the end """
    batchsize, length, classnum = pred.size()
    pred = pred.resize(batchsize * length, classnum)
    label = label.resize(batchsize * length)
    mask = mask.resize(batchsize * length, 1)

    temp = F.log_softmax(pred)
    temp = temp * mask
    loss = F.nll_loss(temp, label, size_average=False)
    loss = loss / (mask.sum(0) + 1e-10)

    return loss


def set_dropout_mode(models, train):
    """ set the mode of all dropout layers """
    for name, model in models._modules.items():
        if model is None:
            continue
        if model.__class__.__name__.find('Dropout') != -1:
            if train:
                model.train()
            else:
                model.eval()
        set_dropout_mode(model, train)

def full_mse_loss(pred, label):
    loss = F.mse_loss(pred, label, size_average=False)
    loss = loss / pred.size(1) / pred.size(0)
    return loss

def full_mse_loss_masked(pred, label, mask):
    loss = torch.sum((pred - label) ** 2 * mask)
    loss = loss / torch.sum(mask)
    return loss
