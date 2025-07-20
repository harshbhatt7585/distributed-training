"""
Understand Gradient Synchronization
1. Topic:
How gradients are averaged across processes in DDP

2. Teach me about the topic:
In DDP, after the backward() call, gradients are automatically averaged across all processes using all_reduce. This ensures that each model replica receives the same gradients.

3. Exercise:

Log the gradients of a parameter (model.layer1.weight.grad) on all ranks after backward pass.

Manually verify that gradients are identical across all ranks.

Add a dist.barrier() to synchronize and print orderly.



"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed as dist

os.environ['NCCL_DEBUG'] = "INFO"
os.environ['NCCL_SOCKET_IFNAME'] = "ens5"

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    return local_rank, global_rank

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        o = self.linear1(x)
        return self.linear2(o)

def train(local_rank):
    print(f"[Process {local_rank}] Starting training")

    model = SimpleModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(local_rank)
            batch_y = batch_y.to(local_rank)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()

            

            grad = model.module.linear1.weight.grad
            dist.barrier()

            print(f"[Rank {local_rank}] Gradient (linear1.weight):\n{grad}")

            optimizer.step()

@record 
def main():
    local_rank, global_rank = ddp_setup()
    train(local_rank)
    destroy_process_group()

if __name__ == '__main__':
    main()
