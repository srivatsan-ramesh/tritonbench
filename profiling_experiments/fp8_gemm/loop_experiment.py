import argparse
import glob
import os

import numpy

import torch

from tritonbench.operators.fp8_gemm.tutorial_profile import (
    matmul_profile as tutorial_matmul_profile,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(script_dir, "dump")
TARGET_DIR = os.path.join(script_dir, "override")

pattern = os.path.join(SOURCE_DIR, "*", "*.ttgir")
file_list = glob.glob(pattern)


def create_overriden_kernel(override_func):
    for file_path in file_list:
        parts = file_path.split(os.sep)

        hash_dir = parts[-2]  # The <hash> part (second to last)
        filename = parts[-1]  # The file name

        target_hash_dir = os.path.join(TARGET_DIR, hash_dir)
        os.makedirs(target_hash_dir, exist_ok=True)
        target_file_path = os.path.join(target_hash_dir, filename)

        with open(file_path, "r") as infile:
            content = infile.read()

        # Insert the additional lines.
        new_content = override_func(content)

        with open(target_file_path, "w") as outfile:
            outfile.write(new_content)


# def insert_warp_group_dot_profile(content):
#     new_content = []
#     for line in content.split('\n'):
#         if re.search("triton_nvidia_gpu\.warp_group_dot_wait.*\{pendings=1\}", line):
#             new_content.append('      tt.proton_record <0, "start", "cycle", "warpgroup">')
#         new_content.append(line)
#         if re.search("triton_nvidia_gpu\.warp_group_dot_wait.*\{pendings=1\}", line):
#             new_content.append('      tt.proton_record <0, "end", "cycle", "warpgroup">')


class RunType:
    NCU = "ncu"
    PASS1 = "pass1"
    PASS2 = "pass2"


def main(m, k, n, num_warps, run_type):
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
    else:
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = f"{script_dir}/dump"
        os.environ["TRITON_KERNEL_OVERRIDE"] = "0"

    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn).T.contiguous().T

    if run_type == RunType.NCU:
        tutorial_matmul_profile(None, a, b, num_warps)
    elif run_type == RunType.PASS2:

        for i in range(100):
            tutorial_matmul_profile(
                f"{script_dir}/traces_pass2/chrome_trace_{i}.json", a, b, num_warps
            )
    else:
        for i in range(100):
            tutorial_matmul_profile(
                f"{script_dir}/traces/chrome_trace_{i}.json", a, b, num_warps
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--pass2", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--num_warps", type=int, default=8)
    args = parser.parse_args()
    main(
        args.m,
        args.k,
        args.n,
        args.num_warps,
        RunType.NCU if args.ncu else RunType.PASS2 if args.pass2 else RunType.PASS1,
    )
