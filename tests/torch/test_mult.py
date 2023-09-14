from stardew.torch import torch_compiler
import pytest
import torch

def test_mult():
    
    vector_size = 128

    a = torch.randint(-10, 10, (vector_size,), dtype=torch.int32)
    b = torch.randint(-10, 10, (vector_size,), dtype=torch.int32)

    @torch_compiler(example=[a,b])
    def mult(a, b):
        return torch.mul(a,b)