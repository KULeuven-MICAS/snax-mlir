import torch
from stardew.torch import torch_compiler


x = torch.randint(-10, 10, (256, 128), dtype=torch.int32)
w = torch.randint(-10, 10, (128,  10), dtype=torch.int32)
b = torch.randint(-10, 10, ( 10,    ), dtype=torch.int32)

@torch_compiler(example=[x, w, b])
def foo(x : torch.Tensor, w : torch.Tensor, b: torch.Tensor):
    return torch.relu(x @ w + b)
