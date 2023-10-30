# %%
# Отчет
# https://wandb.ai/cowboy_bebop/hw2/reports/Failed-HW2--Vmlldzo1ODA4Njcw?accessToken=5uf2d66c61k1ywo2tqdc6hya1x2jnqbhwh3q1s9o14mcyujq37zxsegsmt2g01a4

# %%
# Imports
# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
from torch import nn
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torch.optim.lr_scheduler as lr_scheduler

from random import randint
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# %%
config = {
    "num_classes": 200,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "batch_size": 128,
    "num_epochs": 15,
    "label_smoothing": 0.1,
    "optimizer": torch.optim.AdamW,
    "sheduler": lr_scheduler.StepLR,
    "sheduler_step": 6,
    "sheduler_gamma": 0.8,
    "mean": torch.tensor([0.4802, 0.4481, 0.3976]),
    "std": torch.tensor([0.2770, 0.2691, 0.2821]),
    "label_smooth": 0.1, 
}

# %%
# Custom transfofrms

class MixUp:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x1, x2):
        return x1 * self.alpha + x2 * (1 - self.alpha)
    
class RandomCatOut:
    
    def __init__(self, size, prob):
        self.size = size
        self.prob = prob

    def __call__(self, sample):
        sample = sample.clone().detach()

        pos_x = randint(0, 64 - self.size)
        pos_y = randint(0, 64 - self.size)

        if np.random.binomial(n=1, p=self.prob, size=1)[0] == 1:
            sample[:, pos_x: pos_x + self.size, pos_y: pos_y + self.size] = 0

        return sample
    
class CutMix:
    def __init__(self, size):
        self.size = size

    def __call__(self, x1, x2):
        x1 = x1.clone()
        x2 = x2.clone()

        pos1_x = randint(0, 64 - self.size)
        pos1_y = randint(0, 64 - self.size)

        pos2_x = randint(0, 64 - self.size)
        pos2_y = randint(0, 64 - self.size)

        x1[:, :, pos1_x: pos1_x + self.size, pos1_y: pos1_y + self.size] = \
            x2[:, :, pos2_x: pos2_x + self.size, pos2_y: pos2_y + self.size]
        
        return x1
    
# %%
# Dataloader

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """

    mean = config['mean']
    std = config['std']

    if kind == 'train':
        transform = T.Compose([
            T.ToTensor(),
            T.RandomApply(
                torch.nn.ModuleList([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)]),
                p=0.3
            ),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                torch.nn.ModuleList([T.RandomRotation(degrees=15)]),
                p=0.3
            ),
            T.RandomApply(
                torch.nn.ModuleList([
                    T.RandomCrop(size=48),
                    T.Resize(size=64, antialias=True),
                ]),
                p=0.3
            ),
            RandomCatOut(size=24, prob=0.3),
            T.Normalize(mean, std),
        ])
    else:
        transform = transforms_val = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    folder = ImageFolder(path + kind, transform=transform)

    loader = torch.utils.data.DataLoader(
        folder,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
    )

    return loader

# %%
def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """

    model = torchvision.models.GoogLeNet(
        num_classes=config['num_classes'],
        aux_logits=True,
        dropout_aux=0.2,
    )

    model.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=5, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=5, stride=1)
    )
    model.conv2 = nn.Identity()
    model.conv3 = nn.Identity()
    model.maxpool2 = nn.Identity()

    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    
    optimizer = config["optimizer"](
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )

    return optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """

    model.to('cuda:0')
    prediction = model(batch.to('cuda:0'))
    return prediction

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """

    device = 'cuda:0'
    model.eval()
    model = model.to(device)
    torch.set_grad_enabled(False)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smooth'])
    losses = []

    labels_all = []
    probs_all = []
    preds_all = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        images, labels = batch
        labels.to(device)

        logits = model(images.to(device)).cpu()
        probs = logits.softmax(dim=1)
        max_prob, max_prob_index = torch.max(probs, dim=1)

        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(max_prob.numpy().tolist())
        preds_all.extend(max_prob_index.numpy().tolist())

        loss = loss_fn(logits, labels)

        losses += [loss] * len(labels)

    return accuracy_score(labels_all, preds_all), np.mean(losses)


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """

    device = 'cuda:0'
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smooth'])

    model.to(device)

    for xs, ys_true in tqdm(train_dataloader):

        ys_true = ys_true.to(device)
        output = model(xs.to(device))
        
        o1 = output.aux_logits1.softmax(dim=1)
        o2 = output.aux_logits2.softmax(dim=1)
        o3 = output.logits.softmax(dim=1)


        loss1 = loss_fn(o1, ys_true)
        loss2 = loss_fn(o2, ys_true)
        loss3 = loss_fn(o3, ys_true)

        loss = loss1 * 0.3 + loss2 * 0.3 + loss3 * 0.4

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    with open(checkpoint_path, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(state_dict)

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"

    return "780814f3aabef71a209ec9e5ef2bca83", "https://drive.google.com/file/d/1ZZ6Xn_Zc0HQSy8HJJytvemlgkXbgHhGN/view?usp=sharing"
