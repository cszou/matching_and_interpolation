import torch
from scipy import stats

models = ['m1', 'm2']
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
for k,v in layers.items():
    act_1 = torch.load(f'./activations_results/m1_activations_features_{k}.result.pth.tar')
    act_2 = torch.load(f'./activations_results/m1_activations_features_{k}.result.pth.tar')
    for i in range(layers[k]):
        if i > 64:
            break
        print(f'ranks shape: {act_1.shape}')
        res = stats.kendalltau(act_1[i], act_2[i])
        corr = res.correlation
        print(f'feature{k}_layer{i} correlation: {corr}')