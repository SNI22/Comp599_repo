#!/usr/bin/env python3
import torch
import torch.nn as nn

# 1. Check for GPU
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Define a simple model
model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

# 3. Move model to the device
model.to(device)

# 4. Verify that the model’s parameters are on the correct device
print("Model is on:", next(model.parameters()).device)

# 5. Create a dummy input tensor and move it to device
x = torch.randn(1, 4).to(device)
print("Tensor is on:", x.device)
