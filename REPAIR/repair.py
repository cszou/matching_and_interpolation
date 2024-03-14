import os
from collections import OrderedDict

from torchvision import models, datasets
from tqdm import tqdm
import numpy as np
import scipy.optimize

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision
import torchvision.transforms as T

from weight_matching import weight_interp


traindir = os.path.join('imagenet', 'train')
valdir = os.path.join('imagenet', 'val')
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=256, shuffle=False, num_workers=16, )

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=False, num_workers=16, )

class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def get_stats(self):
        return self.bn.running_mean, self.bn.running_var.sqrt()

    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1


class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


# adds TrackLayers around every conv layer
def make_tracked_net(net):
    net1 = models.alexnet()
    net1.load_state_dict(net.state_dict())
    for i, layer in enumerate(net1.features):
        if isinstance(layer, nn.Conv2d):
            net1.features[i] = TrackLayer(layer)
    return net1.cuda().eval()


# adds ResetLayers around every conv layer
def make_repaired_net(net):
    net1 = models.alexnet()
    net1.load_state_dict(net.state_dict())
    for i, layer in enumerate(net1.features):
        if isinstance(layer, nn.Conv2d):
            net1.features[i] = ResetLayer(layer)
    return net1.cuda().eval()


# reset all tracked BN stats against training data
def reset_bn_stats(model):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()
    model.train()
    with torch.no_grad(), autocast():
        for images, _ in tqdm(train_loader):
            output = model(images.cuda())


def fuse_conv_bn(conv, bn):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 bias=True)

    # set weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused_conv.weight.data = (w_conv * gamma.reshape(-1, 1, 1, 1))

    # set bias
    beta = bn.bias + gamma * (-bn.running_mean + conv.bias)
    fused_conv.bias.data = beta

    return fused_conv


def fuse_tracked_net(net):
    net1 = models.alexnet()
    for i, rlayer in enumerate(net.features):
        if isinstance(rlayer, ResetLayer):
            fused_conv = fuse_conv_bn(rlayer.layer, rlayer.bn)
            net1.features[i].load_state_dict(fused_conv.state_dict())
    net1.classifier.load_state_dict(net.classifier.state_dict())
    return net1


def main():
    p1 = torch.load('./m1.checkpoint.pth.tar')
    p2 = torch.load('./m2.checkpoint.pth.tar')
    m1 = models.alexnet()
    m1.load_state_dict(p1['state_dict'])
    wrap1 = make_tracked_net(m1)
    m2 = models.alexnet()
    m2.load_state_dict(p2['state_dict'])
    wrap2 = make_tracked_net(m2)
    print(wrap1)
    # print(wrap2)

    criterion = CrossEntropyLoss()


    val_model1 = weight_interp.validate(val_loader, m1, criterion)
    val_model2 = weight_interp.validate(val_loader, m2, criterion)
    val_wrap1 = weight_interp.validate(val_loader, wrap1, criterion)
    val_wrap2 = weight_interp.validate(val_loader, wrap2, criterion)
    print('model1:', val_model1)
    print('model2:', val_model2)
    print('wrap1:', val_wrap1)
    print('wrap2:', val_wrap2)
    matchedPara = OrderedDict()
    for k in p1['state_dict'].keys():
        matchedPara[k] = 0.5 * p1['state_dict'][k] + 0.5 * p2['state_dict'][k]
    modelMatched = models.alexnet()
    modelMatched.load_state_dict(matchedPara)
    val_matched = weight_interp.validate(val_loader, modelMatched, criterion)
    print('mixed model:', val_matched)

    reset_bn_stats(wrap1)
    reset_bn_stats(wrap2)
    val_wrap1 = weight_interp.validate(val_loader, wrap1, criterion)
    val_wrap2 = weight_interp.validate(val_loader, wrap2, criterion)
    print('wrap1:', val_wrap1)
    print('wrap2:', val_wrap2)

    corr_vectors = torch.load('./corr.pth.tar')
    alpha = 0.5

    wrap_a = make_repaired_net(modelMatched)
    # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
    # around conv layers in (model0, model1, model_a).
    corr_vec_it = iter(corr_vectors)
    for track0, track1, reset_a in zip(wrap1.modules(), wrap2.modules(), wrap_a.modules()):
        print(track0)
        print(reset_a)
        if not isinstance(track0, TrackLayer):
            continue
        assert (isinstance(track0, TrackLayer)
                and isinstance(track1, TrackLayer)
                and isinstance(reset_a, ResetLayer))

        # get neuronal statistics of original networks
        mu0, std0 = track0.get_stats()
        mu1, std1 = track1.get_stats()
        # set the goal neuronal statistics for the merged network
        goal_mean = (1 - alpha) * mu0 + alpha * mu1
        goal_std = (1 - alpha) * std0 + alpha * std1

        corr_vec = next(corr_vec_it)
        exp_mean = goal_mean
        exp_std = ((1 - alpha) ** 2 * std0 ** 2 + alpha ** 2 * std1 ** 2 + 2 * alpha * (
                    1 - alpha) * std0 * std1 * corr_vec) ** 0.5
        goal_std_ratio = goal_std / exp_std
        goal_mean_shift = goal_mean - goal_std_ratio * exp_mean

        # Y = aX + b, where X has mean/var mu/sigma^2, and we want nu/tau^2,
        # so we set a = tau/sigma and b = nu - (tau / sigma) mu

        reset_a.set_stats(goal_mean_shift, goal_std_ratio)

    model_b = fuse_tracked_net(wrap_a)
    val_wrap_a = weight_interp.validate(val_loader, wrap_a, criterion)
    val_model_b = weight_interp.validate(val_loader, model_b, criterion)
    print(f'wrap a: {val_wrap_a}')
    print(f'model b: {val_model_b}')

if __name__ == '__main__':
    main()