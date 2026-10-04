python run_ner_crf.py --task_name guwen\
    --do_predict_nbest \
    --nbest 5 \
    --data_dir datasets/guwen/ \
    --model_type bert \
    --model_name_or_path guwen_outputs/bert \
    --output_dir guwen_outputs/ \
    --overwrite_output_dir
