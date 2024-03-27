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


all_corr = []
corrs = torch.load('./corr.pth.tar')
for v in corrs:
    all_corr += v.tolist()

para1 = torch.load('./m1.checkpoint.pth.tar')['state_dict']
para2 = torch.load('./m2.checkpoint.pth.tar')['state_dict']
model1 = models.alexnet()
model1.load_state_dict(para1)
model2 = models.alexnet()
model2.load_state_dict(para2)

criterion = nn.CrossEntropyLoss()

matchedPara = OrderedDict()
for k in para1.keys():
    matchedPara[k] = 0.5 * para1[k] + 0.5 * para2[k]
modelMatched = models.alexnet()
modelMatched.load_state_dict(matchedPara)
print('model 1 validation')
val1 = weight_interp.validate(val_loader, model1, criterion)
print('model 2 validation')
val2 = weight_interp.validate(val_loader, model2, criterion)
print('total mix validation')
valM = weight_interp.validate(val_loader, modelMatched, criterion)


val = []
for j in range(0, 10):
    print(f'{10*(10-j)}% mix:')
    threshold = sorted(all_corr)[int(len(all_corr) * j / 10)]
    PartialMatchedPara = OrderedDict()
    for l, k in enumerate(para1.keys()):
        PartialMatchedPara[k] = 0.5 * para1[k] + 0.5 * para2[k]
        if k.split('.')[0] == 'features':
            for i in range(para1[k].shape[0]):
                if corrs[l//2][i] < threshold:
                    PartialMatchedPara[k][i] = para1[k][i]
    PartialModelMatched = models.alexnet()
    PartialModelMatched.load_state_dict(PartialMatchedPara)
    valPartial = weight_interp.validate(val_loader, PartialModelMatched, criterion)
    val.append(valPartial)
    print(val1, val2, valM, valPartial)

torch.save(val, 'corr_val')