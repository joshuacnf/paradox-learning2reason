import argparse
import glob
import json
import logging
import os
import random
from typing import DefaultDict
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import dist
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import random
import pdb
from transformers import AutoTokenizer
from collections import defaultdict

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

 
class LogicDataset(Dataset):
    def __init__(self, examples, args=None, simple_tokenizer_vocab=None):
        self.simple_tokenizer_vocab = simple_tokenizer_vocab
        if args.keep_only_negative:
            self.examples = [i for i in examples if i["label"] == 0]
        self.examples = examples
        for index, example in enumerate(self.examples):
            self.examples[index] = self.convert_raw_example(example)
        
        random.shuffle(self.examples)
        if args.limit_example_num != -1:
            self.examples = self.examples[:args.limit_example_num]

        self.tokenizer  = AutoTokenizer.from_pretrained(
                            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                            do_lower_case=args.do_lower_case,
                            cache_dir=args.cache_dir if args.cache_dir else None,
                        )
        self.max_length = args.max_length
        self.args = args
        if args.skip_long_examples:
            self.skip_long_examples()
        
    def __len__(self):
        return len(self.examples)

    def report_length(self):
        all_leng = []
        print("\n\n")
        total = 200
        for example in self.examples:
            leng = " ".join(example["rules"] + example["facts"]).lower() + ' ' + example["query"].lower()
            
            leng = len(self.tokenizer.tokenize(leng))
            all_leng.append(leng)
            if len(all_leng) == total:
                break
        print("Average_length", sum(all_leng) / total)
        print("Max", max(all_leng))
        print("\n\n")
    
    def report_allkinds_of_stats(self):
        print("\n\n")
        # Number of fact percentage
        all = []
        for example in self.examples:
            all.append(len(example["facts"]) / len(example["preds"]))
        print("Number of fact percentage", sum(all) / len(all))

        # Number of rules percentage
        all = []
        for example in self.examples:
            all.append(len(example["rules"]) / len(example["preds"]))
        print("Number of rules percentage", sum(all) / len(all))

    def convert_raw_example(self, example):
        new_example = {}
        new_example["rules"] = []
        for rule in example["rules"]:
            one_rule = ""
            one_rule +=  " and ".join(rule[0])
            one_rule += ", "
            one_rule += rule[-1]
            one_rule += ' .'
            new_example["rules"].append(one_rule)
        
        new_example["facts"] = []
        for fact in example["facts"]:
            one_fact = "Alice "
            one_fact +=  fact
            one_fact += ""
            new_example["facts"].append(one_fact)
        
        new_example["query"] = "Query: Alice is " + example["query"] + " ?"
        new_example["label"] = example["label"]
        new_example["depth"] = example["depth"]
        new_example["preds"] = example["preds"]
        return new_example


    def __getitem__(self, index):
        
        example = self.examples[index]
        #example = self.convert_raw_example(example)
        
        '''
        "rules": [
        "If Person X is serious, Person Y drop Person X and Person X help Person Y, then Person X get Person Y.",
        "If Person X open Person Y and Person Y help Person X, then Person Y ride Person X."
        ],
        "facts": [
        "Alice is serious.",
        "Alice help Bob.",
        "Bob open Alice."
        ],
        "query": "Alice ride Bob",
        "label": 1
        '''
        if self.args.ignore_fact:
            text_a = " ".join(example["rules"]).lower()
        elif self.args.ignore_both:
            text_a = " "
        else:
            text_a = " ".join(example["rules"] + example["facts"]).lower()
    
        if self.args.ignore_query:
            text_b = " "
        else:
            text_b = example["query"].lower()
        if self.args.shorten_input:
            text_a.strip("If")
            text_a.strip("then")

            text_b.strip("If")
            text_b.strip("then")
        return text_a, text_b, example["label"], example

    def collate_fn(self, examples):
        batch_encoding = self.tokenizer(
            [(example[0], example[1]) for example in examples],
            max_length=self.max_length,
            padding="longest",
            truncation=True)
        if "t5" in self.args.model_name_or_path:
            # encode the label as text
            labels_as_text = ["true" if example[2] == 1 else "false" for example in examples]
            target_encoding = self.tokenizer(labels_as_text, padding="longest", max_length=self.max_length, truncation=True)
            label_ids = torch.tensor(target_encoding.input_ids)
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        else:
            label_ids = torch.LongTensor([example[2] for example in examples])

        return torch.LongTensor(batch_encoding["input_ids"]), torch.LongTensor(batch_encoding["attention_mask"]), torch.LongTensor(batch_encoding["token_type_ids"]) if 'token_type_ids' in batch_encoding else torch.LongTensor([1]), label_ids, [example[-1] for example in examples]
    
    def skip_long_examples(self):
        keep = []
        counter = 0
        for i in tqdm(range(len(self))):
            example = self[i]
            batch_encoding = self.tokenizer(
            [(example[0], example[1])],
            max_length=self.max_length,
            padding="longest",
            truncation=False)
            if len(batch_encoding["input_ids"][0]) > 650:
                print("Over limit")
                counter += 1
            else:
                keep.append(i)
        print("Skipped ", counter, "out of", len(self))
        self.examples = [self.examples[i] for i in keep]


    def limit_length(self, new_length):
        print("Limiting {} to {}".format(len(self), new_length))
        self.examples = self.examples[:new_length]
    
    @staticmethod
    def split_dataset(file_name):
        all_examples = json.load(open(file_name))
        random.seed(0)
        random.shuffle(all_examples)

        train_examples = all_examples[:len(all_examples) // 10 * 8]
        dev_examples = all_examples[len(all_examples) // 10 * 8:len(all_examples) // 10 * 9]
        test_examples = all_examples[len(all_examples) // 10 * 9:]

        with open(file_name + "_train", "w") as f:
            json.dump(train_examples, f)
        with open(file_name + "_val", "w") as f:
            json.dump(dev_examples, f)
        with open(file_name + "_test", "w") as f:
            json.dump(test_examples, f)

        return
    
    @classmethod
    def initialze_from_file(cls, file, args):
        if "," in file:
            files = file.split(",")
        else:
            files = [file]
        all_examples = []
        for file in files:
            with open(file) as f:
                examples = json.load(f)
                all_examples.extend(examples)
        return cls(all_examples, args)
    
    @classmethod
    def initialize_from_file_by_depth(cls, file, args):
        examples_by_depth = cls.load_examples_by_depth(file, depth = args.group_by_which_depth)
        datasets_by_depth = {}
        for depth, _data in examples_by_depth.items():
            datasets_by_depth[depth] = cls(_data, args)

        return datasets_by_depth
    
    @staticmethod
    def load_examples_by_depth(file, depth = "depth"):
        with open(file) as f:
            examples = json.load(f)

        examples_by_depth = defaultdict(list)
        for example in examples:
            examples_by_depth[example[depth]].append(example)
        
        return examples_by_depth

def limit_examples(examples_by_depth, max_depth_during_train, control_num = 2000):

    for key in list(examples_by_depth.keys()):
        if key > max_depth_during_train:
            del examples_by_depth[key]

    limit_length = len(examples_by_depth[max_depth_during_train])
    print("Original lenght", limit_length)
    assert(limit_length >= control_num)
    limit_length = control_num
    print("Limiting to {}".format(limit_length))
    all_examples = []
    for key in examples_by_depth:
        print("Length ", key)
        random.shuffle(examples_by_depth[key]) #.shuffle()
        examples_by_depth[key] = examples_by_depth[key][:limit_length]

        all_examples.extend(examples_by_depth[key])
    return all_examples

def merge_and_balance_dataset(
    file_name, 
    file_range, 
    max_depth_during_train, 
    final_file_name,
    control_num, 
    depth = "depth"):
    all_examples = []
    for i in range(file_range):
        print(i)
        with open(file_name.replace("INDEX", str(i))) as f:
            examples = json.load(f)
    
        examples_by_depth = defaultdict(list)
        for example in examples:
            examples_by_depth[example[depth]].append(example)
        
        all_examples.extend(limit_examples(examples_by_depth, max_depth_during_train, control_num = control_num))

    with open(final_file_name, "w") as f:
        json.dump(all_examples, f)    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="fo_sample/prop_examples.txt")
    parser.add_argument("--file_range", type=int, default=10)
    parser.add_argument("--max_depth_during_train", type=int, default=6)
    parser.add_argument("--final_file_name", type=str, default="fo_sample/prop_examples_balanced.txt")
    parser.add_argument("--depth", type=str, default="depth")
    parser.add_argument("--control_num", type=int, default=2000)
    parser.add_argument("--no_split", action="store_true")
    args = parser.parse_args()

    merge_and_balance_dataset(
        file_name = args.file_name,
        file_range = args.file_range,
        max_depth_during_train = args.max_depth_during_train,
        final_file_name = args.final_file_name,
        control_num = args.control_num,
        depth = args.depth)
    if not args.no_split:
        LogicDataset.split_dataset(args.final_file_name)
