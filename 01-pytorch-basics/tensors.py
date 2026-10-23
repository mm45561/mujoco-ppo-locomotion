import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

print("a + b =", a + b)
print("a * 2 =", a * 2)

# 4 robots, each with 3 joint angles, all starting at 0
angles = torch.zeros(4, 3)
print(angles)
print("shape:", angles.shape)
