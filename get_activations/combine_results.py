import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms


models = [['m1', 'mix'], ['m2', 'm2o']]
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
root = './results'
if not os.path.exists(os.path.join(root, 'combined2')):
    os.makedirs(os.path.join(root, 'combined2'))
    print('folder created!')
else:
    print('folder already exists!')
for layer, channels in layers.items():
    path = os.path.join(root, 'combined2', f'features_{layer}')
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} folder created!')
    else:
        print(f'{path} folder already exists!')
    for c in range(channels):
        fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(20, 12))
        fig.suptitle(f'feature_{layer} channel_{c}', fontsize=25)
        for i in range(2):
            for j in range(2):
                axs[i, j].axis('off')
                axs[i, j].set_title(f'model_{models[i][j]}', fontsize=20)
                img = Image.open(os.path.join(root, models[i][j], f'features_{layer}', f'channel_{c}.png'))
                axs[i, j].imshow(img)
        plt.savefig(os.path.join(path, f'channel_{c}.jpg'))
        plt.close()
    print(f'feature_{layer} done!')
