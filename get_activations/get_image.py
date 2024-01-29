from utils import *


top_indices = torch.load('m1.result.pth.tar')['top_dataset_indices']
standard_test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
imagenet_data = ImageNetWithIndices('/data/imagenet_data', split="train", transform=standard_test_transform)
for k,v in top_indices.items():
    images = {}
    channel = 0
    for top_channel_indices in v.transpose(0,1):
        top_images = list()
        for index in top_channel_indices:
            top_images.append(imagenet_data[index][0])
        images[f'channel_{channel}'] = top_images
        channel += 1
    torch.save(images, f'top_images_{k}')
    print(f'layer_{k} done!')
