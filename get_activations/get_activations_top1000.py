from utils import *
from torch import nn
from tqdm import tqdm
from torchvision import models

if __name__=='__main__':
    k = 1000
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
        top_norms = {}
        top_dataset_indices = {}
        for name, layer in get_model_layers(model).items():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(get_activation(name, activations=activations))
        data_loader = get_topk_dataset_loader(batch_size=1024)
        with torch.no_grad():
            for i, (data, _, batch_indices) in enumerate(tqdm(data_loader)):
                data = data.to(device)
                batch_indices = batch_indices.to(device)
                model(data)
                for key, v in activations.items():
                    if v.shape[0] < 1000:
                        pass
                    activations_norms = torch.linalg.matrix_norm(v)
                    if i == 0:
                        norms, dataset_indices = torch.topk(activations_norms, k, dim=0)
                        # print(norms.shape)
                    else:
                        # For the current batch, get the top norms and their indices
                        batch_norms, b_indices = torch.topk(activations_norms, k=k, dim=0)
                        # Get the dataset indices corresponding to the batch_indices
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
                    'top_dataset_indices': top_dataset_indices}, model_name+'_top1000.result.pth.tar')

