#!/usr/bin/env bash
# ===========================================================
#  共用 NER 训练调用封装
#
#  用法 (先 source):
#    source _train_ner.sh
#
#  函数签名:
#    train_ner <arch> <pretrain> [extra flags...]
#      arch      : crf | softmax
#      pretrain  : sikuroberta | sikubert | bert-chinese-base
#      其余参数 (--do_eval / --do_predict / --do_adv / --data cner ...)
#                 自动透传给 python 脚本
#
#  示例:
#    train_ner crf     sikuroberta
#    train_ner softmax sikuroberta --do_eval --do_predict
#    train_ner crf     sikubert    --do_eval
# ===========================================================

train_ner() {
    local arch="$1"
    local pretrain="$2"
    shift 2                                  # 跳过前两个,剩下的透传

    local py_script="run_ner_${arch}.py"
    local model_path="prev_trained_model/${pretrain}"
    local output_dir="guwen_outputs/${pretrain}"
    local task_name="guwen"

    python "$py_script" \
        --model_type bert \
        --task_name  "$task_name" \
        --data_dir   "datasets/${task_name}/" \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --do_train \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        "$@"
}


train_ner crf  sikuroberta 
