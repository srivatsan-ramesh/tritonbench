import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="fp8_gemm")

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
traces_dir = os.path.join(script_dir, args.op, "traces")

data = {}

for filename in os.listdir(traces_dir):
    with open(os.path.join(traces_dir, filename), "r") as f:
        d = json.load(f)
        for event in d["traceEvents"]:
            if event["pid"] in data:
                data[event["pid"]].add(event["args"]["sm_id"])
            else:
                data[event["pid"]] = {event["args"]["sm_id"]}

with open(os.path.join(script_dir, args.op, "sm_stickiness.txt"), "w") as f:
    for threadblock, sm_ids in data.items():
        if len(sm_ids) > 1:
            f.write(f"Threadblock {threadblock} is running on multiple SMs: {sm_ids}\n")
