from collections import OrderedDict
from torch import nn
from tqdm import tqdm
from torchvision import models
from utils import *


if __name__=='__main__':
    k = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.alexnet()
    model_name = 'mixed_model'
    m1p = torch.load('m1.checkpoint.pth.tar')
    m2p = torch.load('m2.checkpoint.pth.tar')
    mixedModel = OrderedDict()
    # vanillaMatchedModel = OrderedDict()
    for key in m1p['state_dict'].keys():
        mixedModel[key] = 0.5 * m1p['state_dict'][key] + 0.5 * m2p['state_dict'][key]

    model.load_state_dict(mixedModel)

    print(f'Finding top images for {model_name}:')
    activations = {}
    activations_norms = {}
    model.to(device)
    model.eval()
    top_norms = {}
    top_dataset_indices = {}
    for name, layer in get_model_layers(model).items():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(get_activation(name, activations=activations))
    data_loader = get_topk_dataset_loader()
    with torch.no_grad():
        for i, (data, _, batch_indices) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            batch_indices = batch_indices.to(device)
            model(data)
            for key, v in activations.items():
                activations_norms = torch.linalg.matrix_norm(v)
                if i == 0:
                    norms, dataset_indices = torch.topk(activations_norms, k, dim=0)
                else:
                    batch_norms, b_indices = torch.topk(activations_norms, k=k, dim=0)
                    batch_dataset_indices = batch_indices[b_indices].to(device)

                    # Need to stack the indices and norms we already have together, then sort and update the top ones
                    norms_stack = torch.cat((top_norms[key], batch_norms))  # .detach()
                    indices_stack = torch.cat((top_dataset_indices[key], batch_dataset_indices))

                    # get the indices and values of the max norms
                    norms, indices = torch.topk(norms_stack, k=k, dim=0)


                    # gather the dataset indices from the indices stack
                    dataset_indices = torch.gather(indices_stack, 0, indices)
                top_norms[key] = norms
                top_dataset_indices[key] = dataset_indices
    print(f'Top images for {model_name} found!')
    for key, value in top_norms.items():
        print('Layer: ', key)
        print('\tTop image activation norms: ', top_norms[key].shape)
        print('\tTop indices shape : ', top_dataset_indices[key].shape)
    torch.save({'top_norms': top_norms,
                'top_dataset_indices': top_dataset_indices}, model_name+'.result.pth.tar')

