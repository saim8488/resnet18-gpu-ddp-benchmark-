import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # set True later for GPU configs if desired

# ─────────────────────────────────────────────
# 1.  DATASET  (update DATA_ROOT to your path)
# ─────────────────────────────────────────────
TRAIN_ROOT = "./dataset/Training"
TEST_ROOT  = "./dataset/Testing"   # <── change this to where you unzipped the dataset

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(batch_size=32, num_workers=0):
    train_set = datasets.ImageFolder(TRAIN_ROOT, transform=TRAIN_TRANSFORMS)
    test_set  = datasets.ImageFolder(TEST_ROOT,  transform=TEST_TRANSFORMS)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader

# ─────────────────────────────────────────────
# 2.  MODEL
# ─────────────────────────────────────────────
NUM_CLASSES = 4   # glioma, meningioma, pituitary, no tumour

def build_model():
    """ResNet-18 pretrained on ImageNet, head replaced for 4-class output."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# ─────────────────────────────────────────────
# 3.  TRAINING LOOP  (single-process)
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────
# 4.  BENCHMARK RUNNER
# ─────────────────────────────────────────────
def run_experiment(config_name, device_str, use_amp, num_workers,
                   batch_size=32, epochs=10):
    """
    Run one full training experiment and return a metrics dict.

    Args:
        config_name  : human-readable label
        device_str   : 'cpu' or 'cuda'
        use_amp      : bool — enable torch AMP
        num_workers  : int  — DataLoader worker processes
        batch_size   : int
        epochs       : int
    """
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  Device: {device_str} | AMP: {use_amp} | "
          f"Workers: {num_workers} | BS: {batch_size}")
    print(f"{'='*60}")

    device = torch.device(device_str)
    train_loader, test_loader = get_dataloaders(batch_size, num_workers)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device_str) if use_amp else None

    # Reset memory stats
    if device_str == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    loss_history = []   # ADD THIS
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)
        loss_history.append(round(tr_loss, 4))   # ADD THIS
        scheduler.step()
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc:.2f}%  "
              f"time={ep_time:.1f}s")

    total_time = time.perf_counter() - total_start
    _, test_acc = evaluate(model, test_loader, criterion, device)

    n_train = len(train_loader.dataset)
    avg_epoch = sum(epoch_times) / len(epoch_times)
    throughput = (n_train * epochs) / total_time   # images / second

    peak_mem = 0
    if device_str == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    metrics = {
        "config"         : config_name,
        "device"         : device_str,
        "amp"            : use_amp,
        "num_workers"    : num_workers,
        "batch_size"     : batch_size,
        "epochs"         : epochs,
        "loss_history": loss_history,
        "total_time_s"   : round(total_time, 2),
        "avg_epoch_s"    : round(avg_epoch, 2),
        "throughput_img_s": round(throughput, 1),
        "peak_gpu_mem_mb": round(peak_mem, 1),
        "test_acc_pct"   : round(test_acc, 2),
    }

    print(f"\n  ✓ Total time : {total_time:.1f}s")
    print(f"  ✓ Throughput : {throughput:.1f} img/s")
    print(f"  ✓ Peak GPU   : {peak_mem:.1f} MB")
    print(f"  ✓ Test acc   : {test_acc:.2f}%")
    return metrics


# ─────────────────────────────────────────────
# 5.  BATCH-SIZE SWEEP  (Config 4)
# ─────────────────────────────────────────────
def run_batch_sweep(device_str='cuda', use_amp=True, num_workers=2, epochs=5):
    """Run Config 4 across batch sizes [16, 32, 64, 128]."""
    results = []
    for bs in [16, 32, 64, 128]:
        m = run_experiment(
            config_name=f"Config4_BS{bs}",
            device_str=device_str,
            use_amp=use_amp,
            num_workers=num_workers,
            batch_size=bs,
            epochs=epochs,
        )
        results.append(m)
    return results


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--config",      type=str, default="all",
                        choices=["all", "1", "2", "3", "4", "sweep"])
    parser.add_argument("--output",      type=str, default="results.json")
    args = parser.parse_args()

    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        print("⚠  No GPU detected — configs 2–4 will fall back to CPU.")

    gpu = "cuda" if has_gpu else "cpu"
    # Load existing results if they exist
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    if args.config in ("all", "1"):
        all_metrics.append(run_experiment(
            "Config1_CPU",
            device_str="cpu", use_amp=False, num_workers=0,
            batch_size=args.batch_size, epochs=args.epochs,
        ))

    if args.config in ("all", "2"):
        all_metrics.append(run_experiment(
            "Config2_GPU",
            device_str=gpu, use_amp=False, num_workers=0,
            batch_size=args.batch_size, epochs=args.epochs,
        ))

    if args.config in ("all", "3"):
        all_metrics.append(run_experiment(
            "Config3_GPU_AMP",
            device_str=gpu, use_amp=True, num_workers=0,
            batch_size=args.batch_size, epochs=args.epochs,
        ))

    if args.config in ("all", "4"):
        all_metrics.append(run_experiment(
            "Config4_GPU_AMP_ParallelLoad",
            device_str=gpu, use_amp=True, num_workers=2,
            batch_size=args.batch_size, epochs=args.epochs,
        ))

    if args.config == "sweep":
        sweep = run_batch_sweep(device_str=gpu, epochs=5)
        all_metrics.extend(sweep)

    # ── Save results ─────────────────────────
    with open(args.output, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ Results saved → {args.output}")

    # ── Print summary table ──────────────────
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        cpu_time = df.loc[df["config"] == "Config1_CPU", "total_time_s"]
        if not cpu_time.empty:
            df["speedup"] = round(cpu_time.values[0] / df["total_time_s"], 2)
        cols = ["config", "total_time_s", "avg_epoch_s",
                "throughput_img_s", "peak_gpu_mem_mb", "test_acc_pct"]
        if "speedup" in df.columns:
            cols.append("speedup")
        print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
