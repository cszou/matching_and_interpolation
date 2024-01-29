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
    images = []
    for index in top_indices[0]:
        image = dataset[index][0]
        images.append(image)
    return torch.stack(images)

def get_images_from_indices2(indices, num_top_images_per_channel, data_loader=None):
    # Get the dataset object associated with the dataloader
    if data_loader is None:
        dataset = get_topk_dataset_loader().dataset
    else:
        dataset = data_loader.dataset
        # print(indices.shape)

    def grab_image(idx):
        return dataset[idx][0]

    all_images = []
    for nth_place in range(num_top_images_per_channel):
        images = []
        # print(f'nth_place: {nth_place}')
        for index in indices[nth_place]:
            # print(f'index: {index}')
            image = grab_image(index)
            # print('image shape:\n', image.shape)
            images.append(image)
        images_tensor = torch.stack(images)
        all_images.append(images_tensor)
    top_image_tensor = torch.stack(all_images)
    # images = [grab_image(index) for index in indices[0]]
    # print('images from the indices look like:\n', len(images))

    # print('images from the indices look like:\n', top_image_tensor.shape)
    return top_image_tensor


class ImageNetWithIndices(datasets.ImageNet):
    # This makes the dataloader return the dataset indices along with the standard img,label
    # means you can retrieve the original image using data_loader.dataset[index][0]
    # dataset[index] grabs the img,label,index tuple.
    def __getitem__(self, index):
        img, label = super(ImageNetWithIndices, self).__getitem__(index)
        return img, label, index
