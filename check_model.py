import torch

state = torch.load("weights/best.pth", map_location="cpu")
print(type(state))