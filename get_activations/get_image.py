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
for v in top_indices.values():
    print(imagenet_data[v.transpose(0,1)[0][0]])