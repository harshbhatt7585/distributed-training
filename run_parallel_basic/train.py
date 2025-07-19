import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"Rank {rank}/{world_size} using GPU {local_rank}")

    # Dummy training loop
    for i in range(5):
        print(f"[Rank {rank}] Step {i}")
        time.sleep(1)

    dist.destroy_process_group()

if __name__ == "__main__":
    import os
    main()
