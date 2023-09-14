from stardew.torch import torch_compiler
import pytest

def test_mult():
    
    x = torch.randint(-10, 10, (256, 128), dtype=torch.float32)
    w = torch.randint(-10, 10, (128,  10), dtype=torch.float32)
    b = torch.randint(-10, 10, ( 10,    ), dtype=torch.float32)

    @torch_compiler(example=[x, w, b])
    def foo(x : torch.Tensor, w : torch.Tensor, b: torch.Tensor):
        return torch.relu(x @ w + b)