import torch 
from attention import FlashAttention
import time
import psutil

def test_memory_usage():
    attention = FlashAttention(dim=512, heads=8, dim_head=64).cuda()
    x = torch.randn(1, 1000, 512).cuda()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    out = attention(x)
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    print(f"Memory usage: {end_mem - start_mem} bytes")



def test_speed():
    attention = FlashAttention(dim=512, heads=8, dim_head=64).cuda()
    x = torch.cuda(1, 1000, 512).cuda()
    start_time = time.time()

    out = attention(x)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds ")


def test_scalability():
    attention = FlashAttention(dim=512, heads=8, dim_head=64).cuda()
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        x = torch.randn(1, n, 512).cuda()
        start_time = time.time()
        out = attention(x)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Input size: {n} execution time: {end_time - start_time} seconds")


def test_error_rate():
    attention = FlashAttention(dim=512, heads=8, dim_head=64).cuda()
    x = torch.randn(1, 1000, 512).cuda()
    y = torch.randn(1, 1000, 512).cuda()
    out_x = attention(x)
    out_y = attention(y)
    error_rate = (out_x != out_y).float().mean().item()
    print(f"Error rate {error_rate}")


test_memory_usage()
test_speed()
test_scalability()
test_error_rate