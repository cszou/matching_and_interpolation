import torch
from scipy import stats

models = ['m1', 'm2']
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
for k,v in layers.items():
    ranks_1 = torch.load(f'./activations_results/m1_activations_features_{k}.result.pth.tar')
    ranks_2 = torch.load(f'./activations_results/m1_activations_features_{k}.result.pth.tar')
    for i in range(layers[k]):
        if i > 64:
            break
        print(f'ranks shape: {ranks_1.shape}')
        res = stats.kendalltau(ranks_1[i].transpose(0,1), ranks_2[i].transpose(0,1))
        corr = res.correlation
        print(f'feature{k}_layer{i} correlation: {corr}')