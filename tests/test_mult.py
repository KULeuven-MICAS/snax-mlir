import torch
from stardew.torch import torch_compiler


def main():
    vector_size = 128

    a = torch.randint(-10, 10, (vector_size,), dtype=torch.int32)
    b = torch.randint(-10, 10, (vector_size,), dtype=torch.int32)

    @torch_compiler(example=[a, b])
    def foo(a, b):
        return torch.mul(a, b)


def test_mult():
    main()


if __name__ == "__main__":
    main()
