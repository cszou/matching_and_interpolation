import matplotlib.pyplot as plt
import torch


a = torch.load('pic')
plt.imshow(a.permute(1,2,0))
plt.show()
