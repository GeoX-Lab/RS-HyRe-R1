#!/bin/bash
set -e  # 遇到错误立即退出

# 配置区 - 根据实际情况修改
cd /g0001sr/zgz/EasyR1/tests

MODEL_ROOT="/g0001sr/zgz/model/multi_models"  # 模型权重根目录
TEST_SCRIPT="test_ovd.py"                     # 测试脚本名称
OUTPUT_DIR="./checkmulti"                       # 结果输出目录
STEPS=(480)       # 要测试的步长列表（按需调整）
GPU_DEVICES="0,1"                             # 使用的GPU设备

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 遍历所有指定的步长
for STEP in "${STEPS[@]}"; do
    # MODEL_PATH="${MODEL_ROOT}/checknovelty_step_${STEP}"
    # MODEL_PATH="${MODEL_ROOT}/checknovelty_step_${STEP}"
    MODEL_PATH="${MODEL_ROOT}/checkmulti_step_${STEP}"
    
    
    # 检查模型目录是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "警告: 模型目录不存在 - ${MODEL_PATH}, 跳过此步长"
        continue
    fi
    
    echo "=============================="
    echo "正在测试步长: ${STEP}"
    echo "模型路径: ${MODEL_PATH}"
    echo "=============================="
    
    # 临时修改脚本中的关键变量
    TEMP_SCRIPT="ovd_test_step_${STEP}.py"
    cp "$TEST_SCRIPT" "$TEMP_SCRIPT"
    
    # 动态替换模型路径和输出路径
    sed -i "s|MODEL_PATH = .*|MODEL_PATH = \"${MODEL_PATH}\"|" "$TEMP_SCRIPT"
    sed -i "s|OUTPUT_PATH = .*|OUTPUT_PATH = \"${OUTPUT_DIR}/ovd_result_${STEP}.json\"|" "$TEMP_SCRIPT"
    sed -i "s/steps_val = steps/steps_val = ${STEP}/" "$TEMP_SCRIPT"
    
    # 运行测试
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES torchrun --nproc_per_node=2 "$TEMP_SCRIPT"
    
    # 清理临时文件
    rm -f "$TEMP_SCRIPT"
    
    echo "步长 ${STEP} 测试完成! 结果已保存至 ${OUTPUT_DIR}/ovd_result_step${STEP}.json"
    echo
done

echo "所有测试任务已完成!"