#!/bin/bash
cd /home3/nxkh91/HumanML3D/motionfix

# 检查是否已经在运行
if pgrep -f "python3.*autosync.py" > /dev/null; then
    echo "🔄 autosync 已经在运行中 (PID: $(pgrep -f 'python3.*autosync.py'))"
    exit 0
fi

nohup python3 autosync.py > autosync.log 2>&1 &
echo "✅ autosync 已启动 (PID: $!)"
echo "📝 日志: motionfix/autosync.log"
echo "🛑 停止: pkill -f autosync.py"
