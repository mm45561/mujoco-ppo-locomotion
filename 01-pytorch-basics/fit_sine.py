import math

import torch

x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)  # 200 points, shape (200, 1)
y = torch.sin(x)                                         # the curve to learn

model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2001):
    prediction = model(x)
    loss = torch.mean((prediction - y) ** 2)  # average squared error
    optimizer.zero_grad()                     # clear old slopes
    loss.backward()                           # compute new slopes
    optimizer.step()                          # nudge all weights downhill
    if epoch % 250 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.5f}")
