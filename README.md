# On the Paradox of Learning to Reason from Data

This repo provides code for reproducing the experiments in the paper [On the Paradox of Learning to Reason from Data](http://web.cs.ucla.edu/~hzhang19/publication/paradox-learn2reason/paradox-learn2reason.pdf). We provide code for
 - Implementation of a BERT model parameterization which solves SimpleLogic (LogicBERT)
 - Sampling examples from SimpleLogic
 - Training BERT / T5 on SimpleLogic examples

## Environment

Our code primarily uses PyTorch and transformers. For reproducibility, below are the commands we used to setup the environment with docker. However, it should run okay with most versions of Pytorch and transformers.

```
docker run --privileged --name logic --rm -it --runtime=nvidia --ipc=host pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

pip install yacs easydict pillow commentjson attrdict boto3 requests scikit-learn ftfy regex tqdm ml_collections transformers
```

## Eval Logic BERT with Hand-Crafted Parameters
In Section 2.2, we provided a hand-crafted set of parameters for the BERT model (LogicBERT) which solves all examples in SimpleLogic perfectly. We provide an implementation in this repo. To evaluate the model, run the following script.

```
bash scripts/9_eval_logic_bert.sh
```


## Sample Data
To reproduce the dataset we used in the paper, use the following scripts. Note that most of the scripts uses 40 processes. 

#### RP
```
bash 1_generate_rp.bash
```
#### LP
```
bash 2_generate_lp.bash
```

#### LP*
```
bash 3_generate_lp_star.bash
```

#### RP Balanced
```
bash 4_generate_rp_balanced.bash
```



## Train
We trained all models with an effective batch size of 64. The below scripts show how to train BERT / T5 on generated LP data on 4 GPUs.

To train / eval on LP / RP / RP* / RP Balanced, simply specifiy the corresponding ``--train_file_path`` and ``--val_file_path``.

To train on LP + RP, subsample RP and LP data to half of their original size and train on the combined data. E.g.: 

#### BERT

##### Train
```
bash scripts/5_train_bert.bash \
 0,1,2,3 4 9820 \
 OUTPUT/LP/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --train_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_train --val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val
```

##### Evaluation
```
rm eval_result.txt
bash scripts/6_eval_bert.bash 0 \
    --val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val \
    --custom_weight OUTPUT/LP/BERT/random_example_balanced_by_backward_6/checkpoint-19/pytorch_model.bin
cat eval_result.txt
```



#### T5

##### Train

```
bash scripts/7_train_t5.bash \
    0,1,2,3 4 9820 \
    OUTPUT/LP/T5/ \
    --num_train_epochs 20.0 \
    --gradient_accumulation_steps 16 --per_gpu_train_batch_size=1 \
    --train_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_train --val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val
```

##### Evaluation
```
bash scripts/8_eval_t5.bash 0 \
    --val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val \
    --custom_weight OUTPUT/LP/T5/random_example_balanced_by_backward_6/checkpoint-19/pytorch_model.bin
```
