import os
import math
import sys
import json
import random
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_file', default='input.json', type=str)
    arg_parser.add_argument('--output_file', default='output.json', type=str)

    arg_parser.add_argument('--min_rule_num', default=0, type=int)
    arg_parser.add_argument('--max_rule_num', default=80, type=int)

    args = arg_parser.parse_args()

    return args

def stats(examples):
    label_sum = 0.0
    depth_sum = 0.0
    backward_depth_sum = 0.0
    max_tree_depth_sum = 0.0
    tree_depth_sum = 0.0

    example_num = len(examples)
    if example_num == 0:
        return
    for example in examples:
        label_sum += example['label']
        depth_sum += example['depth']
        backward_depth_sum += example['backward_depth']
        max_tree_depth_sum += example['max_tree_depth']
        tree_depth_sum += example['tree_depth']
    print('# of examples:', example_num)
    print('percentage of positive example:', label_sum / example_num)
    print('avg depth:', depth_sum / example_num)
    print('avg backward_depth:', backward_depth_sum / example_num)
    print('avg max_tree_depth:', max_tree_depth_sum / example_num)
    print('avg tree_depth:', tree_depth_sum / example_num)


def main():
    args = init()

    with open(args.input_file, 'r') as fin:
        examples = json.load(fin)
    random.shuffle(examples)
    print("loaded")

    balanced_examples = {}
    for key in range(0, 121):
        balanced_examples[key] = [[], []]

    threshold = 1.0
    for example in examples:
        rule_num = len(example['rules'])
        balanced_examples[rule_num][example['label']].append(example)

    for key in balanced_examples:
        if args.min_rule_num <= key and key <= args.max_rule_num:
            l0 = len(balanced_examples[key][0])
            l1 = len(balanced_examples[key][1])
            threshold = min(threshold, min(l0, l1) * 2.0 / (l0 + l1))

    balanced_examples_ = []
    for key in balanced_examples:
        l0 = len(balanced_examples[key][0])
        l1 = len(balanced_examples[key][1])
        l = math.ceil((l0 + l1) * threshold / 2.0)
        balanced_examples_.extend(balanced_examples[key][0][:l])
        balanced_examples_.extend(balanced_examples[key][1][:l])
        if l0 < l:
            balanced_examples_.extend(balanced_examples[key][1][l:l+l-l0])
        if l1 < l:
            balanced_examples_.extend(balanced_examples[key][0][l:l+l-l1])

    balanced_examples = balanced_examples_

    print(f'threshold: {threshold}')
    print(f'# examples after balance: {len(balanced_examples)}')

    # stats(balanced_examples)

    with open(args.output_file, 'w') as fout:
        json.dump(balanced_examples, fout, indent=2)


if __name__ == '__main__':
    main()