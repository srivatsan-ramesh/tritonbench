import os

script_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = f"{script_dir}/dump"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
os.environ["TRITON_OVERRIDE_DIR"] = f"{script_dir}/override"

import torch

from tritonbench.operators.fp8_gemm.persistent_profile import (
    allocate_matmul_tma,
    matmul_tma_persistent_profile as tutorial_matmul_profile,
)


m = 4096
k = 1024
n = 1024

a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
b = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn).T.contiguous().T

b = b.T.contiguous()
c, desc_a, desc_b, desc_c = allocate_matmul_tma(a, b)

for i in range(100):
    tutorial_matmul_profile(
        f"{script_dir}/traces/chrome_trace_{i}.json", a, b, c, desc_a, desc_b, desc_c
    )
