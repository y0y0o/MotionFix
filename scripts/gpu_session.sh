#!/bin/bash
# Auto-start SLURM GPU interactive session in tmux

SESSION_NAME="gpu-session"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "🖥️  GPU session already running (tmux: $SESSION_NAME)"
    echo "   Attach: tmux attach -t $SESSION_NAME"
    exit 0
fi

# Create a new detached tmux session running srun
tmux new-session -d -s "$SESSION_NAME" \
    "srun -c 1 --gres=gpu:pascal:1 --partition=tpg-gpu-small --time=12:00:00 --pty /bin/bash"

echo "🚀 GPU session requested (tmux: $SESSION_NAME)"
echo "   Attach: tmux attach -t $SESSION_NAME"
echo "   Detach: Ctrl+B, D"
