python run_ner_crf.py --task_name guwen\
    --do_predict \
    --data_dir datasets/guwen/ \
    --model_type bert \
    --model_name_or_path guwen_outputs/sikubert \
    --output_dir guwen_outputs/sikubert/ \
    --overwrite_output_dir 
