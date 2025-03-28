import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="fp8_gemm")

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
traces_dir = os.path.join(script_dir, args.op, "traces")

from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))

regions = {
    "region_0": "triton_nvidia_gpu.warp_group_dot",
    "region_1": "triton_gpu.async_copy_global_to_local",
}

for filename in os.listdir(traces_dir):
    with open(os.path.join(traces_dir, filename), "r") as f:
        d = json.load(f)
        for event in d["traceEvents"]:
            pid = event["pid"]
            tid = event["tid"]
            region = event["name"]

            if (
                isinstance(pid, str)
                and pid.startswith("threadblock")
                and isinstance(tid, str)
                and tid.startswith("warpgroup")
            ):

                key = f"T{pid.split()[-1]}WG{tid.split()[-1]}"
                data[region][key].append(event["dur"])


plt.figure(figsize=(12, 5))

for region, tb_wg_data in data.items():
    combos = []
    variances = []

    for combo, durations in tb_wg_data.items():
        values = np.sum(np.array(durations).reshape(100, -1), axis=1)
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Coefficient of Variation (std / mean) * 100; handle zero-mean case
        if mean_val != 0:
            var_percent = (std_val / mean_val) * 100
        else:
            var_percent = 0.0
        combos.append(combo)
        variances.append(var_percent)

    # Sort for consistent plotting
    combos, variances = zip(*sorted(zip(combos, variances)))

    # Plot
    plt.plot(combos[:64], variances[:64], marker=".", label=regions[region])
plt.xticks(rotation=45)
plt.ylabel("Cycles Variance %", fontsize=20)
plt.xlabel("(M, N, K) = (1024, 1024, 512)", fontsize=20)
plt.title("Cycles Variance % across Threadblocks and Warpgroups")
plt.legend()
plt.tight_layout()

plt.savefig(f"{script_dir}/{args.op}/clock_cycles_variance.png")
