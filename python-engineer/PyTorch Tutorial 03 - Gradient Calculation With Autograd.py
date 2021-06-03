import torch

x = torch.randn(3,requires_grad=True)
print(x)

y = x+2
print(y)

z =y*y*2
z=z.mean()
print(z)
#if z is not a scalar we need to ad tensor to () in backward
z.backward() #dz/dx --- gradient
print(x.grad)

print("------------------------------------")

#########################################

import torch

x = torch.randn(3,requires_grad=True)
print(x)

x.requires_grad_(False)
print(x)

#y=x.detatch
#with torch.no_grad()

############################################
print("------------------------------------")
weights = torch.ones(4, requires_grad=True)
print(weights)
for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)

    weights.grad.zero_() #otherwise gradients will sum up

############################################
print("------------------------------------")