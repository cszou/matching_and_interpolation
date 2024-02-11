import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms


models = [['m1', 'mix'], ['m2', 'm2o']]
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
root = './results'
sims = torch.load('similarities.pth.tar')['similarities']
print(sims)
kts = torch.load('kt.pth.tar')
if not os.path.exists(os.path.join(root, 'combined3')):
    os.makedirs(os.path.join(root, 'combined3'))
    print('folder created!')
else:
    print('folder already exists!')
for layer, channels in layers.items():
    path = os.path.join(root, 'combined3', f'features_{layer}')
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} folder created!')
    else:
        print(f'{path} folder already exists!')
    clips = sims[f'features_{layer}']
    kt = kts[layer]
    for c in range(channels):
        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(20, 12))
        fig.suptitle(f'feature_{layer} channel_{c}\nclip: {clips[c].mean():.3f}, kendall: {kt[c]:.3f}', fontsize=25)
        for i in range(2):
            for j in range(2):
                axs[i, j].axis('off')
                axs[i, j].set_title(f'model_{models[i][j]}', fontsize=20)
                img = Image.open(os.path.join(root, models[i][j], f'features_{layer}', f'channel_{c}.png'))
                axs[i, j].imshow(img)
        plt.savefig(os.path.join(path, f'channel_{c}.jpg'))
        plt.close()
    print(f'feature_{layer} done!')
