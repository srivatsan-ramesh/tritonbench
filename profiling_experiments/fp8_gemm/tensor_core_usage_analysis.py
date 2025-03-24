import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from loop_experiment import main, RunType


script_dir = os.path.dirname(os.path.abspath(__file__))


def get_all_data_as_dict(traces_dir_name="traces"):
    """
    Returns a dictionary of the form: {event_name: list with shape (n_iter, n_tid, n_wid, n_val)}
    """
    traces_dir = os.path.join(script_dir, traces_dir_name)

    data = {}

    for i, filename in enumerate(os.listdir(traces_dir)):
        with open(os.path.join(traces_dir, filename), "r") as f:
            d = json.load(f)
            for event in d["traceEvents"]:
                if event["name"] not in data:
                    data[event["name"]] = []
                if i >= len(data[event["name"]]):
                    data[event["name"]].append([])
                tid = int(event["pid"].split(" ")[1])
                wid = int(event["tid"].split(" ")[1])
                if len(data[event["name"]][i]) <= tid:
                    data[event["name"]][i].extend(
                        [[]] * (tid - len(data[event["name"]][i]) + 1)
                    )
                if len(data[event["name"]][i][tid]) <= wid:
                    data[event["name"]][i][tid].extend(
                        [[]] * (wid - len(data[event["name"]][i][tid]) + 1)
                    )
                data[event["name"]][i][tid][wid].append(event["dur"])
    return data


def get_data_for_region(traces_dir="traces", region_name="region_0"):
    """
    Returns a numpy array of the form (n_iter, n_tid, n_wid, n_val)
    """
    return np.array(get_all_data_as_dict(traces_dir)[region_name])


def get_mean_of_sum(traces_dir="traces", region_name="region_0"):
    """
    Returns a value that sums up the values corresponding to a region and then takes the mean
    """
    return np.mean(np.sum(get_data_for_region(traces_dir, region_name), axis=3))


def _bash(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Automatically decode bytes to str
        shell=True,  # Execute through the shell
        check=True,  # Raise an exception for non-zero return codes
    )
    return result.stdout.strip()


def get_tensor_core_util_from_proton(m, k, n):
    _bash(
        f"python profiling_experiments/fp8_gemm/loop_experiment.py --m {m} --k {k} --n {n}"
    )
    _bash(
        f"python profiling_experiments/fp8_gemm/loop_experiment.py --pass2 --m {m} --k {k} --n {n}"
    )

    return (
        100.0
        * get_mean_of_sum("traces", "region_0")
        / get_mean_of_sum("traces_pass2", "region_0")
    )


def get_tensor_core_util_from_ncu(m, k, n):
    result = _bash(
        f"ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active -k matmul_kernel_profile --csv python profiling_experiments/fp8_gemm/loop_experiment.py --ncu --m {m} --k {k} --n {n}",
    )
    for line in result.split("\n"):
        if "matmul_kernel_profile" in line:
            return float(line.split(",")[-1].strip('"'))
    return 0


SHAPES = [
    (1024, 512, 1024),
    (1024, 1024, 1024),
    (2048, 1024, 64),
    (4096, 1024, 2048),
    (4096, 64, 4096),
]

results = []

for m, k, n in SHAPES:
    results.append(
        [
            get_tensor_core_util_from_ncu(m, k, n),
            get_tensor_core_util_from_proton(m, k, n),
        ]
    )


labels = [f"{m}x{k}x{n}" for m, k, n in SHAPES]
ncu_values = [r[0] for r in results]
ink_values = [r[1] for r in results]

x = np.arange(len(SHAPES))
width = 0.35

# Create a larger figure
fig, ax = plt.subplots(figsize=(8, 6))

rects1 = ax.bar(x - width / 2, ncu_values, width, label="NCU")
rects2 = ax.bar(x + width / 2, ink_values, width, label="InK Prof")

ax.set_ylabel("Utilization (%)")
ax.set_title("Tensor Core Utilization by Shape and Method")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Rotate x labels to prevent overlap
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Optionally, increase y-limit so labels above tall bars don't get cut off
max_val = max(max(ncu_values), max(ink_values))
ax.set_ylim(0, max_val * 1.1)  # 10% headroom above the tallest bar


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects1)
autolabel(rects2)

# Ensure everything fits without cutting off
plt.tight_layout()

warp_suffix = "warp_32"

plt.savefig(f"{script_dir}/tensor_core_usage_{warp_suffix}.png")
with open(f"{script_dir}/util_results/{warp_suffix}.txt", "w") as f:
    f.write(",".join([f"{val:.2f}" for val in ncu_values]))
    f.write("\n")
    f.write(",".join([f"{val:.2f}" for val in ink_values]))
