# FlashAttention2.0: A PyTorch Implementation

FlashAttention is a PyTorch implementation of the Flash Attention mechanism, a memory-efficient and highly parallelizable attention mechanism. This repository provides the code for the Flash Attention module and includes options for parallelization and mixed precision training.

## Installation

To install FlashAttention, you can clone this repository using git:

```bash
git clone https://github.com/yourusername/flashattention.git
cd flashattention
```

Then, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use the FlashAttention module:

```python
import torch
from flashattention import FlashAttention

# Initialize a FlashAttention module
attention = FlashAttention(dim=512, heads=8, dim_head=64)

# Create some random data
x = torch.randn(1, 1000, 512)

# Apply the attention module
out = attention(x)

print(out.shape)  # Outputs: torch.Size([1, 1000, 512])
```

You can also enable parallelization and mixed precision training by setting the `parallel` and `mixed_precision` parameters to `True`:

```python
# Initialize a FlashAttention module with parallelization and mixed precision
attention = FlashAttention(dim=512, heads=8, dim_head=64, parallel=True, mixed_precision=True)

# The rest of the code is the same as before
```

## Tests

Here are some tests to verify the correctness of the forward and backward passes:

```python
import torch
from flashattention import FlashAttention

def test_forward():
    attention = FlashAttention(dim=512, heads=8, dim_head=64)
    x = torch.randn(1, 1000, 512)
    out = attention(x)
    assert out.shape == (1, 1000, 512), f'Unexpected output shape: {out.shape}'

def test_backward():
    attention = FlashAttention(dim=512, heads=8, dim_head=64)
    x = torch.randn(1, 1000, 512, requires_grad=True)
    out = attention(x)
    out.sum().backward()
    assert x.grad is not None, 'No gradient computed'

test_forward()
test_backward()
```

These tests check that the output of the forward pass has the correct shape and that the backward pass correctly computes gradients.
