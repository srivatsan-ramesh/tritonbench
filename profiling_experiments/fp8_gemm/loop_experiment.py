import argparse
import os

import torch

from tritonbench.operators.fp8_gemm.tutorial_profile import (
    matmul_profile as tutorial_matmul_profile,
)

script_dir = os.path.dirname(os.path.abspath(__file__))


class RunType:
    NCU = "ncu"
    PASS1 = "pass1"
    PASS2 = "pass2"


def main(m, k, n, run_type):
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    if run_type == RunType.PASS1:
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = f"{script_dir}/dump"
        os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
        os.environ["TRITON_OVERRIDE_DIR"] = f"{script_dir}/override"
    elif run_type == RunType.PASS2:
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = f"{script_dir}/dump"
        os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
        os.environ["TRITON_OVERRIDE_DIR"] = f"{script_dir}/override_pass2"

    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn).T.contiguous().T

    if args.ncu:
        tutorial_matmul_profile(None, a, b)
    elif args.pass2:
        for i in range(100):
            tutorial_matmul_profile(
                f"{script_dir}/traces_pass2/chrome_trace_{i}.json", a, b
            )
    else:
        for i in range(100):
            tutorial_matmul_profile(f"{script_dir}/traces/chrome_trace_{i}.json", a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--pass2", action="store_true")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()
    main(
        args.m,
        args.k,
        args.n,
        RunType.NCU if args.ncu else RunType.PASS2 if args.pass2 else RunType.PASS1,
    )
