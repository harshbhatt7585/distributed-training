# ddp_debug_test.py
import os
import socket
import torch
import torch.distributed as dist

MASTER_ADDR = "172.31.78.229"
MASTER_PORT = "12355"
RANK = 0 if socket.gethostbyname(socket.gethostname()) == MASTER_ADDR else 1
WORLD_SIZE = 2
LOCAL_RANK = 0  # since each node has 1 GPU

os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = MASTER_PORT
os.environ["RANK"] = str(RANK)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)


# Optional: hardcode NCCL interface for stability
os.environ["NCCL_SOCKET_IFNAME"] = "ens5"
# Avoid unnecessary warnings
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_NET_PLUGIN"] = "none"

print(f"[Node IP={socket.gethostbyname(socket.gethostname())}] assigned RANK={RANK}")

dist.init_process_group(
    backend="nccl",
    rank=RANK,
    world_size=WORLD_SIZE,
)

torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK)

print(f"[Rank {RANK}] CUDA devices: {torch.cuda.device_count()}")
print(f"✅ [Rank {RANK}] Process group initialized.")

tensor = torch.tensor([RANK], device=device, dtype=torch.float32)
print(f"[Rank {RANK}] Initial tensor value: {tensor.item()}")

print(f"[Rank {RANK}] Reached sync point BEFORE all_reduce")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
print(f"[Rank {RANK}] Reached sync point AFTER all_reduce: tensor = {tensor.item()}")

dist.destroy_process_group()
print(f"🔚 [Rank {RANK}] Process group destroyed.")
