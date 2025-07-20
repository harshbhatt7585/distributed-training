"""
DDP Checkpointing and Resuming
1. Topic:
Saving and restoring model and optimizer state in DDP

2. Teach me about the topic:
In DDP, checkpointing must be done carefully:

Save only on rank 0

When resuming, make sure all processes load the same checkpoint

3. Exercise:

Save checkpoints every n epochs

Resume from last checkpoint if it exists

Resume should work from any rank (non-zero as well)

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

def train(local_rank, global_rank):
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
    start_epoch = 0

    if os.path.exists("checkpoint.pt"):
        map_location = f"cuda:{local_rank}"
        checkpoint = torch.load("checkpoint.pt", map_location=map_location)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # start_epoch = checkpoint["epoch"] + 1
        print(f"[Rank {global_rank}] Resumed from checkpoint at epoch {start_epoch}")
    

    for epoch in range(start_epoch, 50):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(local_rank), batch_y.to(local_rank)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            dist.barrier()
            if global_rank == 1:
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, "checkpoint.pt")
            dist.barrier()

            
def main():
    local_rank, global_rank = ddp_setup()
    train(local_rank, global_rank)
    destroy_process_group()

if __name__ == '__main__':
    main()
