#!/bin/bash

# GPU 占用检查函数
check_gpus() {
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    for mem in $usage; do
        if [ "$mem" -ge 100 ]; then
            echo "GPU 占用超过 100MB，当前占用: ${mem}MB"
            return 1
        fi
    done
    return 0
}

# 等待 GPU 空闲
wait_for_gpus() {
    while ! check_gpus; do
        echo "等待 GPU 占用降低..."
        sleep 5  # 每 5 秒检查一次
    done
}

# 带进度条的等待函数（10分钟）
wait_with_progress() {
    local duration=$1  # 总等待时间（秒）
    local interval=1    # 每次更新的间隔时间（秒）
    local total_steps=$((duration / interval))  # 总步数
    local start_time=$(date +%s)  # 开始时间

    echo -n "等待中: ["

    for ((i=0; i<=total_steps; i++)); do
        sleep $interval

        # 计算百分比、经过时间、剩余时间
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))  # 已经过的时间
        local remaining=$((duration - elapsed))  # 剩余的时间
        local progress=$(( (i * 100) / total_steps ))  # 进度百分比

        # 格式化输出进度条
        printf "\r等待中: [%-50s] %d%%  已用: %02d:%02d 剩余: %02d:%02d 总时间: %02d:%02d" \
            "$(printf "%0.s▓" $(seq 1 $((i * 50 / total_steps))))" \
            "$progress" \
            $((elapsed / 60)) $((elapsed % 60)) \
            $((remaining / 60)) $((remaining % 60)) \
            $((duration / 60)) $((duration % 60))
    done

    echo -e "\n完成!"
}


# 启动第一批次任务
run_batch_1() {
    echo "启动批次 1 的任务..."

    tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 80 --downsample_factor_deformation 1 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 40 --downsample_factor_deformation 1 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 1 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 10 --downsample_factor_deformation 1 ; bash" && \

    echo "批次 1 任务已启动！"
}

# 启动第二批次任务
run_batch_2() {
    echo "启动批次 2 的任务..."
    tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 80 --downsample_factor_deformation 2 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 40 --downsample_factor_deformation 2 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 2 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 10 --downsample_factor_deformation 2 ; bash" && \


    echo "批次 2 任务已启动！"
}

# 启动第三批次任务
run_batch_3() {
    echo "启动批次 3 的任务..."
    tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 80 --downsample_factor_deformation 4 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 40 --downsample_factor_deformation 4 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 4 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 4 ; bash" && \

    echo "批次 3 任务已启动！"
}

# 启动第三批次任务
run_batch_4() {
    echo "启动批次 4 的任务..."
    tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 ; bash" && \
    tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerUniform_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 ; bash" && \
    echo "批次 4 任务已启动！"
}

# 启动批次任务并等待
wait_for_gpus
run_batch_1
wait_with_progress 600


wait_for_gpus
run_batch_2
wait_with_progress 600

wait_for_gpus
run_batch_3
wait_with_progress 600

wait_for_gpus
run_batch_4

echo "所有批次任务已启动！"

# i_cmd_scripts/mana_batch.sh