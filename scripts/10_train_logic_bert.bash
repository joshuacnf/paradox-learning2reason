python logic_bert/train_logic_bert.py \
                --data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
                --device cpu --cuda_core 0 --max_epoch 50 \
                --batch_size 4 --lr 0.00001 --weight_decay 0.001 \
                --component_num 10 \
                --log_file log.txt --output_model_file OUTPUT/LP/LOGIC_BERT/ \
                --vocab_file sample/vocab.txt
