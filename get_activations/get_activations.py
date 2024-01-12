import torch
from torchvision import models, datasets, transforms
from torch import nn
from tqdm import tqdm


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


def get_activation(name):
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


class ImageNetWithIndices(datasets.ImageNet):
    # This makes the dataloader return the dataset indices along with the standard img,label
    # means you can retrieve the original image using data_loader.dataset[index][0]
    # dataset[index] grabs the img,label,index tuple.
    def __getitem__(self, index):
        img, label = super(ImageNetWithIndices, self).__getitem__(index)
        return img, label, index


if __name__=='__main__':
    k = 5
    torch.manual_seed(42)
    activations = {}
    activations_norms = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.alexnet()
    model.load_state_dict(torch.load('./checkpoint.pth.tar')['state_dict'])
    model.to(device)
    model.eval()
    data_loader = get_topk_dataset_loader()
    with torch.no_grad():
        for i, (data, _, batch_indices) in enumerate(tqdm(data_loader)):
            for name, layer in get_model_layers(model).items():
                if isinstance(layer, nn.Conv2d):
                    layer.register_forward_hook(get_activation(name))
            data = data.to(device)
            batch_indices = batch_indices.to(device)
            model(data)
            for key, v in activations.items():
                # print(f'v: {v.shape}')
                activations_norms = torch.linalg.matrix_norm(v)
                print(f'activations_norms: {activations_norms.shape}')
                break
            if i == 0:
                top_norms, top_dataset_indices = torch.topk(activations_norms, k, 0)
                print(top_norms.shape)
            else:

                # For the current batch, get the top norms and their indices
                batch_top_norms, batch_top_indices = torch.topk(activations_norms, k=k)
                # Get the dataset indices corresponding to the batch_indices
                batch_top_dataset_indices = batch_indices[batch_top_indices].to(device)


                # Need to stack the indices and norms we already have together, then sort and update the top ones
                norms_stack = torch.cat((top_norms, batch_top_norms))  # .detach()
                indices_stack = torch.cat((top_dataset_indices, batch_top_dataset_indices))

                # get the indices and values of the max norms
                top_norms, top_indices = torch.topk(norms_stack, k=k, dim=-1)


                # gather the dataset indices from the indices stack
                top_dataset_indices = torch.gather(indices_stack, 0, top_indices)
            break
    print('Top images found!')
    print('Top image activation norms:\n', top_norms.shape)
    print('Top indices shape :\n', top_dataset_indices.shape)
    torch.save({'top_norms': top_norms,
                'top_dataset_indices': top_dataset_indices}, 'result.pth.tar')

