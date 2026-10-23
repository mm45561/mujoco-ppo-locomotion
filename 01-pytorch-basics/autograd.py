import torch

x = torch.tensor(3.0, requires_grad=True)  # "track how y depends on x"
y = x ** 2
y.backward()  # compute dy/dx automatically

print("y =", y.item())
print("dy/dx =", x.grad.item())

