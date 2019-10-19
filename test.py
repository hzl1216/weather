import torch
torch2model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
print(torch2model)
