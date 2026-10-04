
train_ner() {
    local arch="$1"
    local pretrain="$2"
    shift 2                                  # 跳过前两个,剩下的透传

    local py_script="run_ner_${arch}.py"
    local model_path="prev_trained_model/${pretrain}"
    local output_dir="cclue_outputs/${pretrain}"
    local task_name="guwen"

    python "$py_script" \
        --model_type bert \
        --task_name  "$task_name" \
        --data_dir   "datasets/cclue/" \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --do_train \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        "$@"
}


# ---- 默认任务:古文 ----
# 训练 + 评估 + 预测同时跑,只需切换下面两行的 arch / pretrain 即可
#
# 改 seed 的三种方式 (优先级:命令行 > 环境变量 > 默认):
#   bash run.sh                  # 默认 --seed 42
#   SEED=7 bash run.sh           # 用环境变量
#   bash run.sh --seed 2024      # 用命令行 (推荐,易复用)
rm -rf datasets/cclue/cached_*
rm -rf cclue_outputs/*
#train_ner crf  bert-base-uncased
train_ner crf     bert-base-chinese
#train_ner crf     sikubert 
#train_ner crf     sikuroberta

