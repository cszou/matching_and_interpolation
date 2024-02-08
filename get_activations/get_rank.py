import numpy as np
from utils import *
from torch import nn
from tqdm import tqdm
from torchvision import models

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.alexnet()
    model_names = {'m1', 'm2'}
    # torch.manual_seed(42)
    for model_name in model_names:
        print(f'Finding top images for {model_name}:')
        activations = {}
        activations_norms = {}
        model.load_state_dict(torch.load(model_name + '.checkpoint.pth.tar')['state_dict'])
        model.to(device)
        model.eval()
        norms = {}
        # ranks = {}
        for name, layer in get_model_layers(model).items():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(get_activation(name, activations=activations))
        data_loader = get_topk_dataset_loader()
        ct = 0
        with torch.no_grad():
            for i, (data, _, batch_indices) in enumerate(tqdm(data_loader)):
                ct += 1
                data = data.to(device)
                batch_indices = batch_indices.to(device)
                model(data)
                for key, v in activations.items():
                    activations_norms = torch.linalg.matrix_norm(v)
                    # print(key, activations_norms.shape)
                    if i == 0:
                        norms[key] = [activations_norms.cpu()]
                    else:
                        norms[key].append(activations_norms.cpu())
                if ct >= 200:
                    break
        for k, v in norms.items():
            rank = np.argsort(torch.cat(v).transpose(0,1).numpy(), 0)
            print(rank.shape)
            # ranks[k] = np.argsort(rank, 0)
            torch.save(rank, model_name+'_ranks' + k +'.result.pth.tar')

