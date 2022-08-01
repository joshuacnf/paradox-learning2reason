CUDA_VISIBLE_DEVICES=$1 python finetune_simplified.py \
        --model_type t5 \
        --tokenizer_name=t5-base \
        --model_name_or_path t5-base \
        --do_eval \
        --do_lower_case \
        --save_steps -1 \
        --per_gpu_eval_batch_size=2   \
        --per_gpu_train_batch_size=2   \
        --overwrite_output_dir \
        --logging_steps 50 \
        --num_workers 1 \
        --warmup_steps 0.05 \
        --max_length 1200 \
        --group_by_which_depth depth \
        --limit_report_max_depth 6 \
        --output_dir OUTPUT/eval \
        ${@:2}