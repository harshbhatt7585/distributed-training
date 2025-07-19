#!/bin/bash

# === CONFIGURATION ===
KEY_PATH="~/harsh-dev-key.pem"                      # 🔁 Your SSH private key
WORKER_IP="172.31.68.158"                           # 🔁 Replace with your worker node IP
REMOTE_USER="ec2-user"
SCRIPT="ddp_debugger.py"
MASTER_PORT=12355
MASTER_IP="172.31.78.229"                           # 🔁 Replace with your master private IP
ENV_ACTIVATE="source /home/ec2-user/miniconda3/bin/activate pytorch"

# === SYNC PYTHON SCRIPT TO WORKER ===
echo "🔁 Syncing $SCRIPT to worker node..."
rsync -avz -e "ssh -i $KEY_PATH" $SCRIPT $REMOTE_USER@$WORKER_IP:~/

# === TEST CONNECTIVITY FIRST ===
echo "🧪 Testing connectivity to worker node..."
if ! ssh -i $KEY_PATH -o ConnectTimeout=10 $REMOTE_USER@$WORKER_IP "echo 'Connection successful'"; then
    echo "❌ Cannot connect to worker node. Check SSH key and IP."
    exit 1
fi

echo "🧪 Testing port connectivity..."
if ! ssh -i $KEY_PATH $REMOTE_USER@$WORKER_IP "nc -z $MASTER_IP $MASTER_PORT"; then
    echo "⚠️  Port $MASTER_PORT is not reachable. This might cause issues."
    echo "   Check security groups allow TCP $MASTER_PORT between instances."
fi

# === START MASTER IN BACKGROUND FIRST ===
echo "🚀 Starting master (node_rank=0) on this machine..."
$ENV_ACTIVATE && \
MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT \
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_IP --master_port=$MASTER_PORT $SCRIPT &
MASTER_PID=$!

# === WAIT FOR MASTER TO BE READY ===
echo "⏳ Waiting for master to initialize..."
sleep 5  # Give master time to start listening

# === CHECK IF MASTER IS STILL RUNNING ===
if ! kill -0 $MASTER_PID 2>/dev/null; then
    echo "❌ Master process died during startup"
    exit 1
fi

# === NOW START WORKER ===
echo "🚀 Starting worker (node_rank=1) on $WORKER_IP..."
ssh -i $KEY_PATH $REMOTE_USER@$WORKER_IP "
  $ENV_ACTIVATE && \
  MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_IP --master_port=$MASTER_PORT $SCRIPT
" &
WORKER_PID=$!

# === WAIT FOR BOTH TO COMPLETE ===
echo "⏳ Waiting for distributed training to complete..."

# Wait for master
wait $MASTER_PID
MASTER_EXIT=$?

# Wait for worker  
wait $WORKER_PID
WORKER_EXIT=$?

# === REPORT RESULTS ===
echo "📊 Results:"
if [ $MASTER_EXIT -eq 0 ]; then
    echo "✅ Master completed successfully"
else
    echo "❌ Master failed with exit code $MASTER_EXIT"
fi

if [ $WORKER_EXIT -eq 0 ]; then
    echo "✅ Worker completed successfully"  
else
    echo "❌ Worker failed with exit code $WORKER_EXIT"
fi

# Exit with error if either failed
if [ $MASTER_EXIT -ne 0 ] || [ $WORKER_EXIT -ne 0 ]; then
    exit 1
fi

echo "🎉 Distributed training completed successfully!"