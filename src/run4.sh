python run_ner_softmax.py --task_name cner \
    --do_train \
    --data_dir datasets/cner/ \
    --model_type bert \
    --model_name_or_path prev_trained_model/bert-base-chinese \
    --output_dir modern_outputs/ \
    --num_train_epochs 5 \
    --overwrite_output_dir 
    #--model_name_or_path prev_trained_model/bert-base-chinese \
