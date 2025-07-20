import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup():
    destroy_process_group()


def save_model(model, rank, path="./fsdp_checkpoint.pt"):
    if rank == 1:
        print(f"[Rank {rank}] Saving checkpoint...")
        torch.save(model.state_dict(), path)


def load_model(model, rank, path="./fsdp_checkpoint.pt"):
    map_location = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    model.load_state_dict(torch.load(path, map_location=map_location))


def train():
    rank, local_rank, world_size = setup()

    model = resnet18()
    model = model.to(local_rank)
    model = FSDP(model, device_id=torch.device(f"cuda:{local_rank}"))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(64, 3, 224, 224)
    y = torch.randint(0, 1000, (64,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16)

    for epoch in range(2):
        for xb, yb in loader:
            xb = xb.cuda(local_rank, non_blocking=True)
            yb = yb.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

        print(f"[Rank {rank}] Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Log memory
        allocated = torch.cuda.memory_allocated(local_rank) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 2)
        print(f"[Rank {rank}] Memory Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    save_model(model, rank)
    load_model(model, rank)

    cleanup()


if __name__ == "__main__":
    train()
