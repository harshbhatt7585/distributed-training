"""
Custom All-Reduce with dist.all_reduce
1. Topic:
Low-level communication primitives

2. Teach me about the topic:
While DDP handles communication automatically, PyTorch exposes low-level APIs like:

dist.all_reduce, dist.broadcast, etc.

These can be used for custom algorithms.

3. Exercise:

Implement manual gradient averaging:

Do loss.backward(), then dist.all_reduce() on each param’s .grad

Divide by world_size

Remove DDP and wrap your own "synchronization logic"
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    return local_rank, global_rank



def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def average_gradients(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            param.grad /= world_size

def train():
    
    local_rank, global_rank = setup()
    print(f"Running on rank {global_rank}")

    torch.manual_seed(0)
    model = SimpleModel().cuda(local_rank)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(64, 10).to(local_rank)
    y = torch.randn(64, 1).to(local_rank)
    dataset = TensorDataset(x, y)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    for epoch in range(3):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()

            average_gradients(model, 2)

            optimizer.step()

        print(f"[Rank {global_rank}] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    train()
