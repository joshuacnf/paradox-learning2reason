mkdir -p $4

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$2 finetune_simplified.py \
    --model_type bert \
    --tokenizer_name=bert-base-uncased \
    --model_name_or_path bert-base-uncased \
    --config_name bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --save_steps -1 \
    --per_gpu_eval_batch_size=1   \
    --per_gpu_train_batch_size=1   \
    --learning_rate 4e-5 \
    --warmup_steps 0.1 \
    --overwrite_output_dir \
    --logging_steps 50 \
    --num_workers 1 \
    --warmup_steps 0.05 \
    --max_length 1000 \
    --seed 10 \
    --output_dir $4 \
    --change_positional_embedding_after_loading \
    ${@:5}


