#!/bin/bash
# 快速测试脚本 - 用于验证环境和方法是否正常工作
# 预计耗时：15-30分钟

set -e

echo "========================================"
echo "快速测试开始"
echo "========================================"

# 1. 检查环境
echo "[1/4] 检查环境..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 检查数据
echo -e "\n[2/4] 检查数据..."
if [ ! -f "data/coco/annotations/instances_val2014.json" ]; then
    echo "错误: 缺少COCO数据，请先运行 bash download_data.sh"
    exit 1
fi

# 3. 快速POPE测试（10样本）
echo -e "\n[3/4] POPE快速测试（10样本，随机分割）..."
python run_pope.py \
    --max_samples 10 \
    --split random \
    --baseline greedy_logits \
    --output_dir results/quick_test/pope

echo -e "\nPOPE测试完成！结果："
cat results/quick_test/pope/*/pope_random.json 2>/dev/null | python -m json.tool | grep -A 5 "metrics"

# 4. 快速CHAIR测试（10样本）
echo -e "\n[4/4] CHAIR快速测试（10样本）..."
python run_chair.py \
    --num_samples 10 \
    --baseline greedy \
    --output_dir results/quick_test/chair

echo -e "\nCHAIR测试完成！结果："
cat results/quick_test/chair/*/chair_summary.json 2>/dev/null | python -m json.tool

echo -e "\n========================================"
echo "快速测试完成！"
echo "如果看到合理的指标输出，说明环境配置正确"
echo "接下来可以运行完整实验："
echo "  - bash reproduce.sh full   # 完整复现"
echo "  - bash reproduce.sh pope   # 只跑POPE"
echo "  - bash reproduce.sh chair  # 只跑CHAIR"
echo "========================================"
