import torch
from torchvision import datasets, transforms


def get_model_layers(model):
    layers = {}
    def get_layers(model, prefix=[]):
        for name, layer in model._modules.items():
            if layer is None:
                continue
            if len(layer._modules) != 0:
                get_layers(layer, prefix=prefix+[name])
            else:
                layers["_".join(prefix+[name])] = layer
    get_layers(model)
    return layers


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def get_topk_dataset_loader(batch_size=256, num_workers=10, imagenet_dir="/data/imagenet_data"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    standard_test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    imagenet_data = ImageNetWithIndices(imagenet_dir, split="train", transform=standard_test_transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader



# When loading in the CLIP model it gives you a preprocess function, pass that here
def get_clip_encoding_dataloader(preprocess, batch_size=256, num_workers=10, imagenet_dir="/data/imagenet_data"):
    imagenet_data = ImageNetWithIndices(imagenet_dir, split="train", transform=preprocess)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader


def get_images_from_indices(top_indices, dataset):
    for k, v in top_indices.items():
        images = {}
        channel = 0
        for top_channel_indices in v.transpose(0, 1):
            top_images = list()
            for index in top_channel_indices:
                top_images.append(dataset[index][0])
            images[f'channel_{channel}'] = top_images
            channel += 1
        torch.save(images, f'top_images_{k}')
        print(f'layer_{k} done!')


class ImageNetWithIndices(datasets.ImageNet):
    # This makes the dataloader return the dataset indices along with the standard img,label
    # means you can retrieve the original image using data_loader.dataset[index][0]
    # dataset[index] grabs the img,label,index tuple.
    def __getitem__(self, index):
        img, label = super(ImageNetWithIndices, self).__getitem__(index)
        return img, label, index
