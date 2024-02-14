import torch
from scipy import stats

models = ['m1', 'm2']
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
results = {}
for k,v in layers.items():
    act_1 = torch.load(f'./activations_results/m1_activations_features_{k}.result.pth.tar')
    act_2 = torch.load(f'./activations_results/m2_activations_features_{k}.result.pth.tar')
    num_channels = act_1.shape[0]
    correlations = torch.zeros([num_channels])
    print(f'features {k}')
    print('correlations shape: ', correlations.shape)
    print('performing kendall tau analysis')
    for i in range(num_channels):
        # print(f'ranks shape: {act_1.shape}')
        res = stats.kendalltau(act_1[i], act_2[i])
        cor = res.correlation
        correlations[i]= cor
        print(cor)
    results[k] = correlations
torch.save(results, 'kt2.pth.tar')