import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="fp8_gemm")
parser.add_argument("--threadblock", type=int, default=0)
parser.add_argument("--warp", type=int, default=0)

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
traces_dir = os.path.join(script_dir, args.op, "traces")

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

warp_group_size = 4
cycles = np.array(data["region_0"])
n_iter, n_tid, n_wid, n_val = cycles.shape

n_warp_groups = n_wid // warp_group_size

cycles_grouped = cycles.reshape(n_iter, n_tid, n_warp_groups, warp_group_size, n_val)

min_per_group = np.min(cycles_grouped, axis=3)

sum_min = np.sum(min_per_group, axis=3)

data_to_plot = []
x_labels = []

for tid in range(n_tid):
    for wg in range(n_warp_groups):
        # For each (tid, warp_group), extract the summed min values across all iterations.
        values = sum_min[:, tid, wg]
        data_to_plot.append(values)
        x_labels.append(f"T{tid}-WG{wg}")

# Create the box-whisker plot using Matplotlib
plt.figure(figsize=(12, 6))
plt.boxplot(
    data_to_plot, patch_artist=True, showfliers=False
)  # patch_artist=True for colored boxes if desired
plt.xticks(range(1, len(x_labels) + 1), x_labels, rotation=45)
plt.xlabel("Thread ID - Warp Group")
plt.ylabel("Tensor Core Usage (cycles)")
plt.title("Box-Whisker Plot for each (tid, warp_group) across Iterations")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{script_dir}/{args.op}/tensor_core_usage.png")
