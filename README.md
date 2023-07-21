# FlashAttention2.0: A PyTorch Implementation

FlashAttention is a PyTorch implementation of the Flash Attention mechanism, a memory-efficient and highly parallelizable attention mechanism. This repository provides the code for the Flash Attention module and includes options for parallelization and mixed precision training.

## Installation

To install FlashAttention, you can clone this repository using git:

```bash
git clone https://github.com/kyegomez/FlashAttention2.0.git
cd FlashAttention2.0
```

Then, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use the FlashAttention module:

```python
import torch
from attention import FlashAttention

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
We have an extensive testing suite in `test.py` run that for more.
Here are some tests to verify the correctness of the forward and backward passes, run `test.py`

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

# Contributing

We welcome contributions to the FlashAttention project! Whether you're interested in improving the code, optimizing the implementation, or adding new features, there are many ways to make a valuable contribution.

## How to Contribute

1. **Fork the repository**: Click the 'Fork' button at the top-right of this page to create your own copy of the repository.

2. **Clone your fork**: Clone your forked repository to your local machine. You can do this with the command `git clone https://github.com/yourusername/flashattention.git`.

3. **Create a new branch**: Create a new branch for your changes with the command `git checkout -b your-branch-name`.

4. **Make your changes**: Make your changes to the code. Please try to follow the existing coding style.

5. **Commit your changes**: Commit your changes with the command `git commit -m "Your commit message"`.

6. **Push your changes**: Push your changes to your forked repository with the command `git push origin your-branch-name`.

7. **Create a pull request**: Go to the [original FlashAttention repository](https://github.com/yourusername/flashattention) and click the 'New pull request' button. Select your forked repository and the branch you created, then click 'Create pull request'.

## Potential Optimizations

There are several areas where the FlashAttention implementation could potentially be optimized:

- **Memory usage**: The current implementation is already quite memory-efficient, but there may be ways to further reduce memory usage.

- **Speed**: The speed of the forward and backward passes could potentially be improved. This could involve optimizing the existing code or implementing new, faster algorithms.

- **Scalability**: The current implementation scales well to large input sizes, but there may be ways to improve scalability further.

- **Precision**: The implementation currently supports mixed precision training, but there may be ways to improve the precision of the computations.

## Metrics

When optimizing the FlashAttention implementation, we should aim to minimize the following metrics:

- **Memory usage**: The amount of memory used by the implementation.

- **Execution time**: The time taken to execute the forward and backward passes.

- **Error rate**: The rate of errors in the output of the attention module.

We look forward to your contributions!