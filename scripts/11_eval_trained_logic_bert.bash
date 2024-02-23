python logic_bert/eval_trained_logic_bert.py \
                --data_file DATA/RP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 1 \
                --batch_size 16 \
                --log_file eval_rp.txt \
                --vocab_file sample/vocab.txt
