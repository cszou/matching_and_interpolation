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
                    # print(key, activations_norms.shape)
                    if i == 0:
                        norms[key] = [activations_norms]
                    else:
                        norms[key].append(activations_norms)
        for k, v in norms.items():
            print(k, torch.cat(v).shape)
        # torch.save({'top_norms': top_norms,
        #             'top_dataset_indices': top_dataset_indices}, model_name+'.result.pth.tar')

