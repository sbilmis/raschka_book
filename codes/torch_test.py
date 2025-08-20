import torch

print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

x = torch.rand(3, 3).to(device)
print(x)
