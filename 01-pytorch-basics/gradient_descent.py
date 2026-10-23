import torch

x = torch.tensor(0.0, requires_grad=True)
lr = 0.1  # learning rate: how big each step is

for i in range(20):
    loss = (x - 5) ** 2          # the error: zero when x is 5
    loss.backward()              # compute the slope
    with torch.no_grad():
        x -= lr * x.grad         # step downhill
    x.grad.zero_()               # clear the slope for the next round
    print(f"step {i:2d}  x={x.item():.3f}  loss={loss.item():.3f}")
