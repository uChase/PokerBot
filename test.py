import torch

if torch.cuda.is_available():
    device = torch.device("cuda")


