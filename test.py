import torch 
from attention import FlashAttention


#init a flash attention module
attention = FlashAttention(dim=512, heads=8, dim_head=64)

#create some random data
x = torch.randn(1, 1000, 512)

#apply the attention module
out = attention(x)

print(out.shape)