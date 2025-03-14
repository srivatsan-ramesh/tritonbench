import os

script_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = f"{script_dir}/dump"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
os.environ["TRITON_OVERRIDE_DIR"] = f"{script_dir}/override"

import torch

from tritonbench.operators.gemm.partition_k_profile import matmul_partition_k


m = 16 * 5
k = 4096 * 9
n = 16 * 5

SPLIT_K_SHAPES = (16 * 5, 16 * 5, 4096 * 9, None)


def _scaled_randn(*args, scale: float, **kwargs) -> torch.Tensor:
    """
    This provides more numerically stable inputs for GEMMs. The +1
    eliminates very small values that could result in denormals, and the
    scale (which should be set to K in an M*N*K GEMM) reduces the size of
    the absolute error.

    In particular, for a given element in the output tensor, the cumulative
    error is eps * 2 * K, where eps is the smallest precision representable
    in the dtype. By scaling the element by K, we avoid the error growing
    with the size of the tensor.
    """
    return (torch.randn(*args, **kwargs) + 1) / scale


a = _scaled_randn((m, k), scale=k, device="cuda", dtype=torch.float16)
b = _scaled_randn((k, n), scale=k, device="cuda", dtype=torch.float16)

for i in range(100):
    matmul_partition_k(f"{script_dir}/traces/chrome_trace_{i}.json", a, b)
