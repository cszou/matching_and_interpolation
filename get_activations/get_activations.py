#!/usr/bin/env python3
import copy

from tqdm import tqdm
import sys

sys.path.insert(0, "/src")
try:
    from utils import *
except:
    from .utils import *
from torchvision import datasets, transforms, models
import torch
import torch.nn.functional as F
# from lucent.optvis import render, param
import os.path as path

# from attack import FastGradientSignUntargeted

def get_attack_activations_function(attack_obj):
    if attack_obj == 'center_neuron':
        print('Using center neuron as attack objective')

        def attack_activation_func(feature_activations):
            shape = feature_activations.shape
            # attack_act = calc_norms(feature_activations[:, channels, shape[2] // 2, shape[3] // 2])
            attack_act = feature_activations[:, :, shape[2] // 2, shape[3] // 2].square()
            return attack_act
    elif attack_obj == 'channel':
        print('Using channel as attack objective')

        def attack_activation_func(feature_activations):
            attack_act = calc_norms_by_channel(feature_activations)
            return attack_act
    else:
        print("ATTACK_OBJ NOT SUPPORTED. DEFAULTING TO CHANNEL")

        def attack_activation_func(feature_activations):
            attack_act = calc_norms_by_channel(feature_activations)
            return attack_act
    return attack_activation_func

def calc_norms_by_channel(activations):
    # print('activations shape:\n', activations.shape)
    flattened = activations.flatten(start_dim=2)
    norms = flattened.square().mean(dim=2)

    # print('flat shape:\n', flattened.shape)
    # test_flatten = activations.flatten(start_dim=-2)
    # print('flat2 shape:\n', test_flatten.shape)
    # print(flattened)

    # print('norms shape:\n', norms.shape)
    # norms2 = flattened.square().mean(dim=-1)
    # print('norms shape:\n', norms2.shape)
    return norms


def get_topk_image_indices_by_channel(model, activations_dict, k, get_attack_activations, imagenet_folder,
                                      data_loader=None):
    # If no images are requested, return without doing anything else
    if k == 0:
        return
    with torch.no_grad():
        if data_loader is None:
            print('getting the dataloader inside the get_topk function')
            data_loader = get_topk_dataset_loader(imagenet_dir=imagenet_folder)
        model.eval()

        print('Beginning top image search!')
        for i, (data, _, batch_indices) in enumerate(tqdm(data_loader)):

            # print('batch indices\n', batch_indices.shape)
            # print(batch_indices[0:10])
            # print('dataset[0]\n', len(data_loader.dataset[0]))
            # print('dataset[1] and dataset[2]\n', data_loader.dataset[0][1], data_loader.dataset[0][2])

            # exit()
            data = data.to(_default_device)

            batch_indices = batch_indices.to(_default_device)
            # Populate the activations dict
            model(data)
            # Array of (channels, topk_indices). Need norm and index of the corresponding image
            # track index using i and batch_size

            # make a vector with the norms associated with each image we got activations for
            activations = activations_dict[FEATURE_NAME]

            # returns a (batch,channels) activation tensor
            activ_norms = get_attack_activations(activations)

            if i == 0:  # First batch setup

                # Grab the top indices for k entries in the norms
                # In general, dataset indices indicate that they are for accessing the data through the dataset object.
                # In the first case they are the same as the indices of the batch itself.
                # Later they will have to be tracked separately
                top_norms, top_dataset_indices = torch.topk(activ_norms, k=k, dim=0)

            else:

                # For the current batch, get the top norms and their indices
                batch_top_norms, batch_top_indices = torch.topk(activ_norms, k=k, dim=0)
                # Get the dataset indices corresponding to the batch_indices
                batch_top_dataset_indices = batch_indices[batch_top_indices].to(_default_device)


                # Need to stack the indices and norms we already have together, then sort and update the top ones
                norms_stack = torch.cat((top_norms, batch_top_norms))  # .detach()
                indices_stack = torch.cat((top_dataset_indices, batch_top_dataset_indices))

                # get the indices and values of the max norms
                top_norms, top_indices = torch.topk(norms_stack, k=k, dim=0)


                # gather the dataset indices from the indices stack
                top_dataset_indices = torch.gather(indices_stack, 0, top_indices)
        print('Top images found!')
        print('Top image activation norms:\n', top_norms.shape)
        print('Top indices shape :\n', top_dataset_indices.shape)
        return top_dataset_indices


def get_topk_dataset_loader(batch_size=256, num_workers=10, imagenet_dir="/data/imagenet_data/train"):
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


class ImageNetWithIndices(datasets.ImageNet):
    # This makes the dataloader return the dataset indices along with the standard img,label
    # means you can retrieve the original image using data_loader.dataset[index][0]
    # dataset[index] grabs the img,label,index tuple.
    def __getitem__(self, index):
        img, label = super(ImageNetWithIndices, self).__getitem__(index)
        return img, label, index


def register_hooks(model_layers, features, optim_layer):
    def get_features(name):
        def hook(model, input, output):
            features[name] = output  # .detach()

        return hook

    model_layers[optim_layer[0]].register_forward_hook(get_features(FEATURE_NAME))


if __name__ == "__main__":
    _default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_NAME = 'layer_features'
    model = models.alexnet()
    model.load_state_dict(torch.load('../results/model1_orig.checkpoint.pth.tar'))
    activations_dict = {}
    get_attack_activations = get_attack_activations_function('channel')
    # Define the function for the hook.
    model_layers = get_model_layers(model)
    # print(model_layers, feature_layer, channels)

    register_hooks(model_layers, activations_dict, feature_layer)
    get_topk_image_indices_by_channel(model=model, activations_dict=activations_dict, k=10, get_attack_activations=get_attack_activations, imagenet_folder='/data/imagenet_data/train',
                                      data_loader=None)


