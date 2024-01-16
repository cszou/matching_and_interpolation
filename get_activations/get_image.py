import torch
from torchvision import datasets, transforms


class ImageNetWithIndices(datasets.ImageNet):
    # This makes the dataloader return the dataset indices along with the standard img,label
    # means you can retrieve the original image using data_loader.dataset[index][0]
    # dataset[index] grabs the img,label,index tuple.
    def __getitem__(self, index):
        img, label = super(ImageNetWithIndices, self).__getitem__(index)
        return img, label, index



top_indices = torch.load('m1.result.pth.tar')['top_dataset_indices']
standard_test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
imagenet_data = ImageNetWithIndices('/data/imagenet_data', split="train", transform=standard_test_transform)
images = {}
for k,v in top_indices.items():
    images[k] = {}
    channel = 0
    for top_channel_indices in v.transpose(0,1):
        top_images = list()
        for index in top_channel_indices:
            top_images.append = imagenet_data[index][0]
        images[k][f'channel_{channel}'] = top_images
        channel += 1

torch.save(images, 'top_images')
