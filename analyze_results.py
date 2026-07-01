import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
with open("results.json") as f:
    data = json.load(f)

try:
    with open("results_ddp.json") as f:
        ddp = json.load(f)
    data.append(ddp)
except FileNotFoundError:
    print("results_ddp.json not found — skipping Config 5.")

df = pd.DataFrame(data)

# Compute speedup relative to CPU baseline
cpu_time = df.loc[df["config"].str.contains("CPU"), "total_time_s"].values
if len(cpu_time):
    df["speedup"] = round(cpu_time[0] / df["total_time_s"], 2)
else:
    df["speedup"] = 1.0

df.to_csv("results_summary.csv", index=False)
print(df[["config", "total_time_s", "throughput_img_s",
          "peak_gpu_mem_mb", "test_acc_pct", "speedup"]].to_string(index=False))


# ─────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
LABEL_MAP = {
    "Config1_CPU"                  : "Config 1\nCPU",
    "Config2_GPU"                  : "Config 2\nGPU",
    "Config3_GPU_AMP"              : "Config 3\nGPU+AMP",
    "Config4_GPU_AMP_ParallelLoad" : "Config 4\nGPU+AMP+DL",
    "Config5_DDP"                  : "Config 5\nDDP",
}

def nice_labels(series):
    return [LABEL_MAP.get(v, v) for v in series]

def bar_chart(ax, x_labels, values, ylabel, title, colors=COLORS):
    bars = ax.bar(x_labels, values, color=colors[:len(values)],
                  edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    return ax


# ─────────────────────────────────────────────
# 3.  MAIN CONFIG COMPARISON CHARTS
# ─────────────────────────────────────────────
main_df = df[df["config"].isin(LABEL_MAP.keys())].copy()
labels  = nice_labels(main_df["config"])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("PDC Training Benchmark — ResNet-18 on Brain Tumour MRI\n(23i-2065 | AI-B)",
             fontsize=14, fontweight="bold", y=1.01)

bar_chart(axes[0,0], labels, main_df["total_time_s"].tolist(),
          "Total Training Time (s)", "A  Total Training Time")

bar_chart(axes[0,1], labels, main_df["throughput_img_s"].tolist(),
          "Throughput (images / second)", "B  Training Throughput")

bar_chart(axes[1,0], labels, main_df["peak_gpu_mem_mb"].tolist(),
          "Peak GPU Memory (MB)", "C  Peak GPU Memory Usage")

bar_chart(axes[1,1], labels, main_df["test_acc_pct"].tolist(),
          "Test Accuracy (%)", "D  Final Test Accuracy")

plt.tight_layout()
plt.savefig("figures/benchmark_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ figures/benchmark_overview.png")


# ─────────────────────────────────────────────
# 4.  SPEEDUP CHART
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
bar_chart(ax, labels, main_df["speedup"].tolist(),
          "Speedup (×) over CPU Baseline", "Speedup over CPU Baseline")
ax.axhline(1.0, color="red", linestyle="--", linewidth=0.9, label="CPU baseline (1×)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/speedup_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ figures/speedup_bar.png")


# ─────────────────────────────────────────────
# 5.  BATCH-SIZE SWEEP  (Config 4 variants)
# ─────────────────────────────────────────────
sweep_df = df[df["config"].str.startswith("Config4_BS")].copy()
if not sweep_df.empty:
    sweep_df = sweep_df.sort_values("batch_size")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Config 4 — Batch Size Sweep (GPU + AMP + Parallel DataLoader)",
                 fontsize=12, fontweight="bold")

    bs_labels = [f"BS={v}" for v in sweep_df["batch_size"]]
    bar_chart(axes[0], bs_labels, sweep_df["throughput_img_s"].tolist(),
              "Throughput (images / second)", "Throughput vs Batch Size",
              colors=["#55A868"]*len(sweep_df))
    bar_chart(axes[1], bs_labels, sweep_df["peak_gpu_mem_mb"].tolist(),
              "Peak GPU Memory (MB)", "GPU Memory vs Batch Size",
              colors=["#C44E52"]*len(sweep_df))

    plt.tight_layout()
    plt.savefig("figures/batch_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ figures/batch_sweep.png")
else:
    print("  (No batch-sweep data found — run: python train.py --config sweep)")

# Loss curves
fig, ax = plt.subplots(figsize=(9, 5))
for i, row in main_df.iterrows():
    if "loss_history" in row and isinstance(row["loss_history"], list):
        label = LABEL_MAP.get(row["config"], row["config"])
        ax.plot(range(1, len(row["loss_history"]) + 1),
                row["loss_history"], marker="o", label=label)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Curves — All Configurations", fontweight="bold")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ figures/loss_curves.png")

# Scaling efficiency
if "Config5_DDP" in df["config"].values and "Config2_GPU" in df["config"].values:
    gpu_time  = df.loc[df["config"] == "Config2_GPU", "total_time_s"].values[0]
    ddp_time  = df.loc[df["config"] == "Config5_DDP", "total_time_s"].values[0]
    ddp_speedup    = round(gpu_time / ddp_time, 2)
    ddp_efficiency = round((ddp_speedup / 2) * 100, 1)
    print(f"\nDDP Scaling Efficiency: {ddp_efficiency}%  (speedup={ddp_speedup}x over single GPU)")
print("\n✅ All figures saved to ./figures/")
print("✅ Summary table saved → results_summary.csv")
