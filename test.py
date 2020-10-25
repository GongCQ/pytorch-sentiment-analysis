import torch
import torch.nn as nn

loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.Tensor([0, 2, 1]) # torch.empty(3).random_(2)
output = loss(input, target)
output.backward()