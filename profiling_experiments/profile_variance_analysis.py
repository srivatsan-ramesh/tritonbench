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

for filename in os.listdir(traces_dir):
    with open(os.path.join(traces_dir, filename), "r") as f:
        d = json.load(f)
        for event in d["traceEvents"]:
            if (
                event["pid"] == f"threadblock {args.threadblock}"
                and event["tid"] == f"warp {args.warp}"
            ):
                if event["name"] in data:
                    data[event["name"]].append(event["dur"])
                else:
                    data[event["name"]] = [event["dur"]]


plt.figure(figsize=(10, 6))

region_names = list(data.keys())
y_positions = range(len(region_names))

for i, region in enumerate(region_names):
    values = np.array(data[region])
    # Scatter plot of raw data with vertical jitter to avoid overlapping exactly on the same y value.
    jitter = np.random.uniform(-0.1, 0.1, size=values.shape)
    y_jitter = i + jitter
    plt.scatter(values, y_jitter, alpha=0.7, label=region)

    # Compute p50 (median) and p99 percentiles
    p50 = np.percentile(values, 50)
    p99 = np.percentile(values, 99)
    plt.scatter(
        p50, i, marker="D", color="red", s=100, label="P50 (Median)" if i == 0 else ""
    )
    plt.scatter(p99, i, marker="s", color="blue", s=100, label="P99" if i == 0 else "")

    # Compute mean and standard deviation (for error bars)
    mean_val = np.mean(values)
    std_val = np.std(values)
    plt.errorbar(
        mean_val,
        i,
        xerr=std_val,
        fmt="o",
        color="black",
        markersize=10,
        capsize=25,
        markeredgewidth=2,
        alpha=0.4,
        label="Mean ± STD" if i == 0 else "",
    )

plt.xlabel("Clock Cycles")
plt.ylabel("Region")
plt.title(
    "Scatter Plot of Clock Cycles with Percentiles and Mean ± STD for Each Region"
)
plt.yticks(list(y_positions), region_names)
plt.grid(True)
plt.legend()

# Save the plot to a file instead of showing it.
plt.savefig(f"{script_dir}/{args.op}/clock_cycles_scatter.png")
