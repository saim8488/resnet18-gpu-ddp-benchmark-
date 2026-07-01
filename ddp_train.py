import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models

# ─────────────────────────────────────────────
# CONFIG  (keep consistent with train.py)
# ─────────────────────────────────────────────
TRAIN_ROOT = "./dataset/Training"
TEST_ROOT  = "./dataset/Testing" 
NUM_CLASSES = 4
SEED        = 42
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 1e-3

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# SETUP / TEARDOWN
# ─────────────────────────────────────────────
def setup(rank=None, world_size=None):
    """Initialise the default process group.
    If rank/world_size provided, use them (spawn mode).
    Otherwise use env vars set by torchrun.
    """
    if rank is not None and world_size is not None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo")   # use "nccl" if multi-GPU


def cleanup():
    dist.destroy_process_group()


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def get_ddp_loaders(rank, world_size, batch_size=32, num_workers=0):
    train_set = datasets.ImageFolder(TRAIN_ROOT, transform=TRAIN_TRANSFORMS)
    test_set  = datasets.ImageFolder(TEST_ROOT,  transform=TEST_TRANSFORMS)

    # Use DistributedSampler for training data
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False,  # shuffle=False when using sampler
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader, train_sampler

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, sampler, criterion, optimizer, device, epoch):
    model.train()
    sampler.set_epoch(epoch)       # ensures different shuffling per epoch
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()            # gradients averaged across processes by DDP
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
# MAIN
# ─────────────────────────────────────────────
def main(rank=None, world_size=None):
    # Support both spawn mode (rank/world_size passed) and torchrun mode
    if rank is not None and world_size is not None:
        setup(rank, world_size)
    else:
        setup()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    is_master  = (rank == 0)

    # Simulate multi-GPU by mapping each process to the same device
    # On a real multi-GPU machine replace with: device = torch.device(f"cuda:{rank}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_master:
        print(f"\n{'='*60}")
        print(f"  Config 5 — DDP  |  world_size={world_size}")
        print(f"{'='*60}")

    torch.manual_seed(SEED)
    train_loader, test_loader, train_sampler = get_ddp_loaders(rank, world_size)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Wrap model for distributed training
    model = DDP(model, device_ids=None)  # device_ids=None works for CPU or shared GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, train_sampler,
            criterion, optimizer, device, epoch
        )
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)
        scheduler.step()

        if is_master:
            print(f"  Epoch {epoch:>2}/{EPOCHS}  "
                  f"loss={tr_loss:.4f}  acc={tr_acc:.2f}%  "
                  f"time={ep_time:.1f}s")

    total_time = time.perf_counter() - total_start

    # Only rank 0 evaluates and saves metrics
    if is_master:
        _, test_acc = evaluate(model, test_loader, criterion, device)
        n_train = len(train_loader.dataset)
        throughput = (n_train * EPOCHS) / total_time
        peak_mem = 0
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        metrics = {
            "config"          : "Config5_DDP",
            "world_size"      : world_size,
            "total_time_s"    : round(total_time, 2),
            "avg_epoch_s"     : round(sum(epoch_times) / len(epoch_times), 2),
            "throughput_img_s": round(throughput, 1),
            "peak_gpu_mem_mb" : round(peak_mem, 1),
            "test_acc_pct"    : round(test_acc, 2),
        }

        print(f"\n  ✓ Total time : {total_time:.1f}s")
        print(f"  ✓ Throughput : {throughput:.1f} img/s")
        print(f"  ✓ Peak GPU   : {peak_mem:.1f} MB")
        print(f"  ✓ Test acc   : {test_acc:.2f}%")

        with open("results_ddp.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("✅ Results saved → results_ddp.json")

    cleanup()


if __name__ == "__main__":
    # Check if launched via torchrun (env vars present)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        main()
    else:
        # Spawn mode for Windows compatibility (no torchrun needed)
        # Use WORLD_SIZE=1 to avoid memory issues on single GPU
        WORLD_SIZE = 1
        print(f"Launching {WORLD_SIZE} process(es) via mp.spawn...")
        mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
