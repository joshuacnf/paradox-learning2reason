CUDA_VISIBLE_DEVICES=$1 python finetune_simplified.py \
    --model_type bert \
    --tokenizer_name=bert-base-uncased \
    --model_name_or_path bert-base-uncased \
    --do_eval \
    --do_lower_case \
    --save_steps -1 \
    --per_gpu_eval_batch_size=2   \
    --per_gpu_train_batch_size=2   \
    --overwrite_output_dir \
    --num_workers 1 \
    --max_length 1000 \
    --output_dir OUTPUT/eval \
    --group_by_which_depth depth \
    --limit_report_max_depth 6 \
    --change_positional_embedding_before_loading \
    ${@:2}
