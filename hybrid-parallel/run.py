import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'  # or master IP in multi-node
    # os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Model Parallel Toy Model: part A on cuda:rank, part B on cuda:(rank+1)%2
class ModelParallelToyModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.device_a = torch.device(f'cuda:{rank}')
        self.device_b = torch.device(f'cuda:{(rank + 1) % torch.cuda.device_count()}')

        self.net_a = nn.Linear(512, 1024).to(self.device_a)
        self.relu = nn.ReLU()
        self.net_b = nn.Linear(1024, 10).to(self.device_b)

    def forward(self, x):
        x = x.to(self.device_a)
        x = self.relu(self.net_a(x))
        x = x.to(self.device_b)
        return self.net_b(x)

def train(rank, world_size):
    setup(rank, world_size)

    model = ModelParallelToyModel(rank)
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    inputs = torch.randn(64, 512)
    targets = torch.randint(0, 10, (64,))

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets.to(outputs.device))
        loss.backward()
        optimizer.step()

        print(f"[Rank {rank}] Epoch {epoch+1} | Loss: {loss.item():.4f}")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            print(f"[Rank {rank}] Device {i}: Allocated={allocated:.2f} MB | Reserved={reserved:.2f} MB")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
