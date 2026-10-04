python run_ner_crf.py --task_name guwen\
    --do_train \
    --data_dir datasets/guwen/ \
    --model_type bert \
    --model_name_or_path prev_trained_model/sikuroberta \
    --output_dir guwen_outputs/ \
    --num_train_epochs 5 \
    --overwrite_output_dir 
    #--model_name_or_path prev_trained_model/bert-base-chinese \
