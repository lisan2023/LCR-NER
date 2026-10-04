python run_ner_crf.py --task_name guwen\
    --do_train \
    --data_dir datasets/baihua/ \
    --model_type bert \
    --model_name_or_path prev_trained_model/sikuroberta \
    --num_train_epochs 5 \
    --train_max_seq_length 256 \
    --output_dir baihua_outputs/ \
    --overwrite_output_dir 
