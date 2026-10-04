#!/usr/bin/env bash
# ===========================================================
#  共用 NER 训练调用的封装函数
#
#  用法 (先 source):
#    source _train_ner.sh
#
#  函数签名:
#    train_ner <python_script> <task_name> <data_dir>
#              <model_path> <output_dir> [extra flags...]
#
#  默认参数: --do_train --num_train_epochs 5 --overwrite_output_dir
#  其余参数 (--do_eval/--do_predict/--do_adv/...) 自动透传
#
#  示例:
#    train_ner run_ner_crf.py     guwen  datasets/guwen/  prev_trained_model/bert-base-chinese  guwen_outputs/
#    train_ner run_ner_softmax.py cner # 1. 先训练(用 run2.sh)
sh run2.sh

# 2. 再跑 n-best 解码
sh run_nbest.sh

# 输出长这样:
guwen_outputs/test_nbest.json    ← 每句 5 条候选 + score + prob
  datasets/cner/   prev_trained_model/bert-base-chinese  cner_outputs/   --do_eval
#    train_ner run_ner_crf.py     guwen  datasets/guwen/  prev_trained_model/sikuroberta        guwen_outputs/ --do_eval --do_predict
# ===========================================================

train_ner() {
    local py_script="$1"
    local task_name="$2"
    local data_dir="$3"
    local model_path="$4"
    local output_dir="$5"
    shift 5

    python "$py_script" \
        --model_type bert \
        --task_name  "$task_name" \
        --data_dir   "$data_dir" \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --do_train \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        "$@"
}
