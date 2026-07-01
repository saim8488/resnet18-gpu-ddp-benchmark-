import json
import numpy as np

with open("runs_breakdown.json") as f:
    runs = json.load(f)

# Load existing results.json to preserve sweep configs (BS16, BS32 etc)
with open("results.json") as f:
    existing = json.load(f)

# Keep only the sweep entries (they don't have multiple runs)
sweep_configs = [r for r in existing if r["config"].startswith("Config4_BS")]

metrics_to_avg = ["total_time_s", "throughput_img_s", "peak_gpu_mem_mb", "test_acc_pct"]

averaged = []
for config_name, data in runs.items():
    if config_name == "note":
        continue
    
    entry = {"config": config_name}
    for metric in metrics_to_avg:
        values = [data[f"run{i}"][metric] for i in range(1, 4)]
        entry[metric] = round(float(np.mean(values)), 2)
    
    averaged.append(entry)

# Combine averaged configs + sweep configs
final = averaged + sweep_configs

with open("results.json", "w") as f:
    json.dump(final, f, indent=2)

print("✅ results.json updated with averaged values")