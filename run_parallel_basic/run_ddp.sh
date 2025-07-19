#!/bin/bash

# === CONFIGURATION ===
KEY_PATH="~/harsh-dev-key.pem"                 # 🔁 Your SSH private key (.pem)
WORKER_IP="172.31.68.158"                 # 🔁 Replace with your WORKER private IP
REMOTE_USER="ec2-user"                    # For Amazon Linux
SCRIPT="train.py"                         # Script to run
MASTER_PORT=12355
ENV_ACTIVATE="source /home/ec2-user/miniconda3/bin/activate pytorch"  # Your conda env

# === DETECT MASTER PRIVATE IP ===
MASTER_IP="172.31.78.229"
echo "👑 Master IP: $MASTER_IP"

# === SYNC SCRIPT TO WORKER ===
echo "🔁 Syncing $SCRIPT to worker node..."
rsync -avz -e "ssh -i $KEY_PATH" $SCRIPT $REMOTE_USER@$WORKER_IP:~/

# === START WORKER VIA SSH ===
echo "🚀 Starting worker (node_rank=1) on $WORKER_IP..."

ssh -i $KEY_PATH $REMOTE_USER@$WORKER_IP "
  $ENV_ACTIVATE && \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_IP --master_port=$MASTER_PORT $SCRIPT
" &
WORKER_PID=$!

# === START MASTER LOCALLY ===
echo "🚀 Starting master (node_rank=0) on this machine..."
$ENV_ACTIVATE && \
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_IP --master_port=$MASTER_PORT $SCRIPT

# Optional: wait for both to finish
wait $WORKER_PID
