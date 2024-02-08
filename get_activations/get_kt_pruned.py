import torch
from scipy import stats

models = ['m1', 'm2']
layers = {'0': 64, '3': 192, '6': 384, '8': 256, '10': 256}
for k,v in layers.items():
    ranks_1 = torch.load(f'm1_ranks_features_{k}.result.pth.tar')
    ranks_2 = torch.load(f'm2_ranks_features_{k}.result.pth.tar')
    num_channels = ranks_1.shape[0]
    correlations = torch.zeros([num_channels])
    print('correlations shape: ', correlations.shape)
    print('performing kendall tau analysis')

    count = 0

    for i in range(num_channels):
        channel_1 = ranks_1[i]
        mask_1 = channel_1 > 0.1 * channel_1.max()
        mask_1_indices = mask_1.nonzero().squeeze()
        channel_2 = ranks_2[i]
        mask_2 = channel_2 > 0.1 * channel_2.max()
        mask_2_indices = (mask_2).nonzero().squeeze()

        mask = (torch.cat((mask_1_indices, mask_2_indices))).unique()
        channel_1_pruned = channel_1[mask].cpu().numpy()
        channel_2_pruned = channel_2[mask].cpu().numpy()

        res = stats.kendalltau(channel_1_pruned, channel_2_pruned)
        cor = res.correlation
        correlations[i]= cor
        print(corr)