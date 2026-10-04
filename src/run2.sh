python run_ner_crf.py --task_name guwen\
    --do_train \
    --data_dir datasets/guwen/ \
    --model_type bert \
    --model_name_or_path prev_trained_model/bert-base-chinese \
    --num_train_epochs 5 \
    --output_dir guwen_outputs/ \
    --overwrite_output_dir 
