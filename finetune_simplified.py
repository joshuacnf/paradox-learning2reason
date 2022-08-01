# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
import argparse
import glob
import json
import logging
import os
import random
import pprint
from typing import DefaultDict
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import dist
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics as compute_metrics,
    glue_convert_examples_to_features as convert_examples_to_features,
    glue_output_modes as output_modes,
    glue_processors as processors,
)
import pdb
from transformers import BertForSequenceClassification
from helpers import *

from dataset import LogicDataset

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn = train_dataset.collate_fn, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # resume_dir
    if args.resume_dir is not None:
        print("Resume training from: ", args.resume_dir)
        if not args.resume_dir.endswith("--1"):
            args.model_name_or_path = args.resume_dir
            print("Load Model Weight")
            model.load_state_dict(torch.load(args.resume_dir + "/pytorch_model.bin", map_location="cpu"))

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_warmup_steps = int(t_total * args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        print("Loading optimizer and scheduler from checkpoints", args.model_name_or_path)
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location="cpu"))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location="cpu"))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        #set global_step to global_step of last saved checkpoint from model path
        epochs_trained = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained += 1
        global_step = epochs_trained * len(train_dataloader) * args.gradient_accumulation_steps
        print("  Continuing training from checkpoint, will skip to saved global_step")
        print("  Continuing training starting from epoch %d", epochs_trained)
        print("  Continuing training from global step %d", global_step)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    train_meter = TrainingMeter()
    epoch_num = epochs_trained

    for _ in train_iterator:
        epoch_iterator = train_dataloader
        for step, batch in enumerate(tqdm(epoch_iterator)):
            batch, examples = batch[:-1], batch[-1]
            if args.skip_training:
                break

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert" and args.model_type != "t5":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            
            with torch.cuda.amp.autocast(enabled=args.use_autocast):
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            logits = outputs[1]
            labels = batch[3]

            acc = (logits.argmax(-1) == labels).sum().float() / labels.view(-1).size(0)
            train_meter.update(
                {
                    "loss": loss.item(),
                    "acc": acc.item()
                }
            )
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):  
                if not args.fp16 and not args.use_autocast:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}, **{"step_per_epoch": len(train_dataloader) // args.gradient_accumulation_steps}}))
                    train_meter.report()
                    train_meter.clean()
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to %s", output_dir)
                
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.local_rank <= 0:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(epoch_num))
            print("Saving model checkpoint to ", output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            # save optimizer
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # save scheduler
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        evaluate(args, model, tokenizer, eval_dataset=eval_dataset)

        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", eval_dataset = None):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn = eval_dataset.collate_fn, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    output_strings = []
    label_strings = []

    results_by_reasoning_depth = defaultdict(int)
    counter_by_reasoning_depth = defaultdict(int)
    def nested_defaultdict():
        return defaultdict(int)

    label_distribution_by_reasoning_depth = defaultdict(nested_defaultdict)

    correct_or_not_all = defaultdict(list)
    correct_counter = 0
    total_counter = 0
    for _, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        model.eval()
        batch, examples = batch[:-1], batch[-1]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert" and args.model_type != "t5":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if args.model_type == "t5":
                # if model has module
                if hasattr(model, "module"):
                    _module = model.module
                else:
                    _module = model
                output_sequences = _module.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        do_sample=False,  # disable sampling to test if batching affects output
                )
                outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                output_strings.extend(outputs)
                label_strings.extend(tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True))
                nb_eval_steps += 1
            else:
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        if args.model_type == "t5":
            correct_or_not = [output_strings[i] == label_strings[i] for i in range(len(output_strings))]
            correct_counter += sum(correct_or_not)
            total_counter += len(correct_or_not)
        
        if args.report_example_length:
            
            correct_or_not = (logits.argmax(-1) == inputs["labels"].detach()).cpu().tolist()

            for index in range(len(examples)):
                results_by_reasoning_depth[examples[index]["depth"]] += correct_or_not[index]
                counter_by_reasoning_depth[examples[index]["depth"]] += 1
                label_distribution_by_reasoning_depth[examples[index]["depth"]][examples[index]["label"]] += 1
            
            for index in range(len(examples)):
                correct_or_not_all[examples[index]["example_index"]].append(correct_or_not[index])

    if args.report_example_length:
        print()
        keys = list(results_by_reasoning_depth.keys())
        keys.sort()
        for key in keys:
            if args.local_rank <= 0:
                print("    Depth {}: {}".format(key, results_by_reasoning_depth[key]/counter_by_reasoning_depth[key]))
                print("        Label_distribution {} : {}".format(key, label_distribution_by_reasoning_depth[key]))
    
    if "t5" in args.model_name_or_path:
        result = {"acc": correct_counter / total_counter}
        results.update(result)
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = {"acc": (out_label_ids == preds).mean()}
        
        results.update(result)

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        #help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        #help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_visualization", action="store_true")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--from_scratch", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--nopooler", action="store_true", help="Do not load the pooler",
    )
    parser.add_argument("--seed", type=int, default=9595, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--custom_weight",
        type=str,
        default=None
    )
    parser.add_argument(
        "--custom_config",
        type=str,
        default=None
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--file_root", type=str, default=None)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--start_gradual_index", type=int, default=1)
    parser.add_argument("--load_bert_weight", type=str, default=None)


    parser.add_argument("--use_gradual_sampler", action="store_true")
    parser.add_argument('--limit_to_negative_examples', action="store_true")
    parser.add_argument('--limit_to_positive_examples', action="store_true")

    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--further_split", action="store_true")
    parser.add_argument("--further_further_split", action="store_true")
    parser.add_argument("--report_example_length", action="store_true")
    parser.add_argument("--ignore_fact", action="store_true")
    parser.add_argument("--ignore_both", action="store_true")
    parser.add_argument("--ignore_query", action="store_true")
    parser.add_argument("--change_positional_embedding_after_loading", action="store_true")
    parser.add_argument("--change_positional_embedding_before_loading", action="store_true")

    parser.add_argument("--shorten_input", action="store_true")
    parser.add_argument('--shrink_ratio', default=1, type=int)
    parser.add_argument('--use_autocast', action="store_true")

    parser.add_argument('--max_depth_during_train', default=1000, type=int)

    parser.add_argument("--train_file_path", default=None)
    parser.add_argument("--val_file_path", default=None)
    parser.add_argument("--group_by_which_depth", default="depth")
    parser.add_argument("--keep_only_negative", action="store_true")
    parser.add_argument("--limit_report_depth", default=-1, type=int)
    parser.add_argument("--limit_report_max_depth", default=100, type=int)
    parser.add_argument("--skip_long_examples", action="store_true")
    parser.add_argument("--limit_example_num", default=-1, type=int)
    parser.add_argument("--resume_dir", default=None)


    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    setup_for_distributed(args.local_rank <= 0)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    num_labels = 2

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if "bert" in args.model_name_or_path: 
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    if "t5" in args.model_name_or_path:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    if args.change_positional_embedding_before_loading:
        expand_position_embeddings(model, args.max_length, args.model_name_or_path)

    if args.custom_weight is not None:
        model.apply(model._init_weights)
        custom_state_dict = torch.load(args.custom_weight, map_location='cpu')
        for key in list(custom_state_dict.keys()):
            custom_state_dict[key.replace("module.", "")] = custom_state_dict[key]
        load_state_dict_flexible(model, custom_state_dict)
        print("\n\nLoaded {}".format(args.custom_weight))

    if args.load_bert_weight is not None:
        original_bert_weight = torch.load(args.load_bert_weight, map_location="cpu")
        old_keys = []
        new_keys = []
        for key in original_bert_weight.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            original_bert_weight[new_key] = original_bert_weight.pop(old_key)
        load_state_dict_flexible(model, original_bert_weight)

    if args.change_positional_embedding_after_loading:
        expand_position_embeddings(model, args.max_length, args.model_name_or_path)

    if args.nopooler:
        model.bert.pooler.apply(model._init_weights)

    if args.from_scratch:
        print("\n\nReinitializing parameters\n\n")
        model.bert.apply(model._init_weights)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    print("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = LogicDataset.initialze_from_file(args.train_file_path, args)
        train_dataset.report_length()
        
        val_dataset = LogicDataset.initialze_from_file(args.val_file_path, args)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, val_dataset)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        print("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    # Evaluation
    print("Enterring evaluation")
    
    if args.do_eval and args.local_rank in [-1, 0]:
        model.eval()
        if "," in args.val_file_path:
            val_files = args.val_file_path.split(",")
        else:
            val_files = [args.val_file_path]
        
        all_results = {}
        all_kinds_of_results = []
        results_string_final = ""
        for val_file in val_files:
            results_string = {}
            results = []
            print("\n\n", val_file)
            val_dataset = LogicDataset.initialze_from_file(val_file, args)
            val_dataset.report_allkinds_of_stats()

            datasets = LogicDataset.initialize_from_file_by_depth(val_file, args)
            depths = list(datasets.keys())
            depths.sort()
            total_example = sum([len(datasets[i]) for i in datasets])
            for depth in depths:
                print("\n\n")
                print("Evaluating examples of depth ", depth)
                result = evaluate(args, model, tokenizer, eval_dataset=datasets[depth])
                results_string[depth] = "Acc: {} ; Percentage {}".format(result["acc"], len(datasets[depth]) / total_example)
                all_kinds_of_results.append(result["acc"])
                if depth >= args.limit_report_depth and depth <= args.limit_report_max_depth:
                    results.append(result['acc'])
            
            pprint.pprint(results_string)
            results_string_final += val_file + "\n\n"
            results_string_final += pprint.pformat(results_string)
            results_string_final += "\n\n\n"

            all_kinds_of_results.insert(0, sum(all_kinds_of_results) / len(all_kinds_of_results))
        
            all_results[val_file] = "{:.3f}".format((sum(results) / len(results)) * 100)
            all_kinds_of_results.insert(1, sum(results) / len(results))
        print("Final Reporting")

        for key in sorted(list(all_results.keys())):
            print(key)
        print()
        for key in sorted(list(all_results.keys())):
            print(all_results[key])
        
        pprint.pprint(all_results)

        with open("eval_result.txt", "a+") as f:
            f.write(args.custom_weight)
            f.write("\n\n")
            f.write(results_string_final)
            f.write("\n\n\n\n\n")


if __name__ == "__main__":
    main()