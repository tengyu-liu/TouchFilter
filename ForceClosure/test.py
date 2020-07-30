import torch
torch.manual_seed(0)
print(torch.rand(1, device='cuda'))