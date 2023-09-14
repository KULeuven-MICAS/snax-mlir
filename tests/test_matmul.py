import torch
from stardew.torch import torch_compiler


def main():
    # These should ideally also be converted to integer datatypes,
    # but torch-mlir does not support non floating-point types for
    # the RELU activation function

    x = torch.randint(-10, 10, (256, 128), dtype=torch.float32)
    w = torch.randint(-10, 10, (128, 10), dtype=torch.float32)
    b = torch.randint(-10, 10, (10,), dtype=torch.float32)

    @torch_compiler(example=[x, w, b])
    def foo(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        return torch.relu(x @ w + b)


def test_matmul():
    main()


if __name__ == "__main__":
    main()
