import os
from collections import OrderedDict

from torchvision import models, datasets

import torch
from torch import nn
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


all_kts = []
kts = torch.load('../get_activations/kt2.pth.tar')
for v in kts.values():
    all_kts += v.tolist()

threshold = sorted(all_kts)[int(len(all_kts)*0.2)]

para1 = torch.load('../get_activations/m1.checkpoint.pth.tar')['state_dict']
para2 = torch.load('../get_activations/m2.checkpoint.pth.tar')['state_dict']
model1 = models.alexnet()
model1.load_state_dict(para1['state_dict'])
model2 = models.alexnet()
model2.load_state_dict(para2['state_dict'])

criterion = nn.CrossEntropyLoss()

matchedPara = OrderedDict()
for k in para1.keys():
    print(k)
    matchedPara[k] = 0.5 * para1[k] + 0.5 * para2[k]
    if k.split('.')[0] == 'features':
        for i in range(para1[k].shape[0]):
            if kts[k.split('.')[1]][i] < threshold:
                matchedPara[k][i] = para1[k][i]
modelMatched = models.alexnet()
modelMatched.load_state_dict(matchedPara)


val1 = weight_interp.validate(val_loader, model1, criterion)
val2 = weight_interp.validate(val_loader, model2, criterion)
valM = weight_interp.validate(val_loader, modelMatched, criterion)
print(val1, val2, valM)