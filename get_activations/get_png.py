import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from utils import *

top_indices = torch.load('m1.result.pth.tar')['top_dataset_indices']
standard_test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
imagenet_data = ImageNetWithIndices('/data/imagenet_data', split="train", transform=standard_test_transform)
for k,v in top_indices.items():
    images = {}
    v = v.transpose(0,1)
    if not os.path.exists(f'./results/m1/{k}'):
        os.makedirs(f'./results/m1/{k}')
        print(f'folder: {k} created')
    else:
        print(f'folder: {k} already exists')
    for channel in range(v.shape[0]):
        top_channel_indices = v[channel]
        plt.figure(figsize=(10,4), frameon=False)
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.axis('off')
            idx = top_channel_indices[i]
            plt.imshow(imagenet_data[idx][0].permute(1,2,0))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'./results/m1/{k}/channel_{channel}.png')
        plt.close()
    print(f'layer_{k} done!')
