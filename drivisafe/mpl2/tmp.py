import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = (108, 192),
    patch_size = 6,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 192, 108)

preds = v(img) # (1, 1000)
print(img.shape)
print(preds.shape)