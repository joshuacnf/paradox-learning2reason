import os
import sys
import json
import random
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    arg_parser.add_argument('--output_file', default='prop_examples.txt', type=str)

    arg_parser.add_argument('--example_num', default=1000, type=int)
    arg_parser.add_argument('--min_pred_num', default=5, type=int)
    arg_parser.add_argument('--max_pred_num', default=30, type=int)

    arg_parser.add_argument('--balance_by_depth', action='store_true')
    arg_parser.add_argument('--max_depth', default=6, type=int)

    arg_parser.add_argument('--algo', default='RP', type=str)

    args = arg_parser.parse_args()

    return args


def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r') as fin:
        vocab = [line.strip() for line in fin.readlines()]
    print('vocabulary size: ', len(vocab))
    return vocab


def sample_one_rule(preds):
    head_num = random.randint(1, 3)
    lits = random.sample(preds, min(head_num + 1, len(preds)))
    random.shuffle(lits)
    return (lits[:-1], lits[-1])


def sample_rule_priority(preds):
    pred_num = len(preds)
    rule_num = random.randint(0, 4 * pred_num)
    fact_num = random.randint(0, pred_num)

    cache = set()
    rules = []
    for _ in range(0, rule_num):
        rule = None
        while True:
            rule = sample_one_rule(preds)
            rule_hash = ' '.join(sorted(rule[0])) + ' ' + rule[1]
            if rule_hash not in cache:
                cache.add(rule_hash)
                break
        rules.append(rule)

    facts = random.sample(preds, fact_num)

    query = random.sample(preds, 1)[0]

    return rules, facts, query


def sample_label_priority(preds):
    preds_ = preds[:]
    random.shuffle(preds_)
    pred_num = len(preds)

    graph_depth = random.randint(1, pred_num // 2)
    width = pred_num // graph_depth

    preds_0 = preds_[:pred_num % graph_depth]
    preds_ = preds_[pred_num % graph_depth:]

    rules = []
    levels = []

    prev_level = [[x, random.randint(0, 1)] for x in preds_[:width]]
    if graph_depth > 1:
        prev_level[0][1], prev_level[1][1] = 0, 1
    else:
        prev_level[0][1], prev_level[1][1], prev_level[2][1], prev_level[3][1] = 0, 1, 0, 1
    preds_ = preds_[width:]
    levels.append(prev_level)

    # phase_1
    for d in range(0, graph_depth - 1):
        level = [[x, random.randint(0, 1)] for x in preds_[:width]]
        preds_ = preds_[width:]
        if len(preds_0) != 0:
            level.append((preds_0[0], random.randint(0, 1)))
            preds_0 = preds_0[1:]
        level[0][1], level[1][1] = 0, 1

        for node in level:
            lit, label = node[0], node[1]
            head_cand = [x[0] for x in prev_level if x[1] == label]
            head_num = random.randint(1, min(3, len(head_cand)))
            head = random.sample(head_cand, head_num)
            rules.append((head, lit))

        levels.append(level)
        prev_level = level

    # phase_2
    rule_num = random.randint(0 * pred_num, 3 * pred_num)
    nodes = [x for y in levels for x in y]
    neg_nodes = [x for x in nodes if x[1] == 0]
    rule_cnt = 0
    while rule_cnt < rule_num:
        tail_node = random.sample(nodes, 1)[0]

        tail = tail_node[0]
        head_cand = [x for x in nodes if x[0] != tail]
        while True:
            head_num = random.randint(1, min(3, len(head_cand)))
            head_nodes = None
            head_nodes = random.sample(head_cand, head_num)
            if not (all([x[1] == 1 for x in head_nodes]) and tail_node[1] == 0):
                break
        head = [x[0] for x in head_nodes]
        rules.append((head, tail))
        rule_cnt += 1

        # if all predicates in the head and tail of a rule are True,
        # we add one extra rule where its head and tail are both False
        # to balance the number of True/Positive predicates in all rules
        if all(x[1] == 1 for x in head_nodes):
            neg_tail = random.sample(neg_nodes, 1)[0][0]
            neg_head_cand = [x for x in neg_nodes if x[0] != neg_tail]
            neg_head_num = random.randint(1, min(3, len(neg_head_cand)))
            neg_head_nodes = random.sample(neg_head_cand, neg_head_num)
            neg_head = [x[0] for x in neg_head_nodes]
            rules.append((neg_head, neg_tail))
            rule_cnt += 1

    facts = [x[0] for x in levels[0] if x[1] == 1]

    query = random.sample([x[0] for x in nodes], 1)[0]

    return rules, facts, query


def sample_lp_star(preds):
    preds_ = preds[:]
    pred_num = len(preds)

    graph_depth = random.randint(2, pred_num // 2)
    width = pred_num // graph_depth

    preds_0 = preds_[:pred_num % graph_depth]
    preds_ = preds_[pred_num % graph_depth:]

    rules = []
    levels = []

    prev_level = [[x, random.randint(0, 1)] for x in preds_[:width]]
    prev_level[0][1], prev_level[1][1] = 0, 1
    preds_ = preds_[width:]
    levels.append(prev_level)

    # phase_1
    for d in range(0, graph_depth - 1):
        level = [[x, random.randint(0, 1)] for x in preds_[:width]]
        if preds_0 != []:
            level.append((preds_0[0], random.randint(0, 1)))
            preds_0 = preds_0[1:]
        level[0][1], level[1][1] = 0, 1
        preds_ = preds_[width:]

        for node in level:
            lit, label = node[0], node[1]
            head_nodes_cand = prev_level
            if label == 1:
                head_nodes_cand = [x for x in prev_level if x[1] == 1]
            head_num = random.randint(1, min(3, len(head_nodes_cand)))
            while True:
                head_nodes = random.sample(head_nodes_cand, head_num)
                if not (all([x[1] for x in head_nodes]) and label == 0):
                    break
            head = [x[0] for x in head_nodes]
            rules.append((head, lit))

        levels.append(level)
        prev_level = level

    # phase_2
    rule_num = random.randint(0 * pred_num, 3 * pred_num)
    nodes = [x for y in levels for x in y]
    for _ in range(0, rule_num):
        tail_d = random.randint(0, len(levels) - 2)
        tail_level = levels[tail_d]
        tail_node = random.sample([x for x in tail_level if x[1] == 1], 1)[0]

        tail = tail_node[0]
        head_cand = [x for y in levels[tail_d:] for x in y
            if x[0] != tail]
        head_num = random.randint(1, min(3, len(head_cand)))
        while True:
            head_nodes = random.sample(head_cand, head_num)
            if not all([x[1] for x in head_nodes]):
                break
        head_nodes = random.sample(head_cand, head_num)
        head = [x[0] for x in head_nodes]
        rules.append((head, tail))

    facts = [x[0] for x in levels[0] if x[1] == 1]

    query = random.sample([x[0] for x in nodes], 1)[0]

    return rules, facts, query


def forward_chain(rules, facts):
    res = {}
    for fact in facts:
        res[fact] = 0

    depth = 1
    prev_len = 0
    while len(res) > prev_len:
        new_facts = []
        for rule in rules:
            head, tail = rule
            if all([lit in res for lit in head]):
                new_facts.append(tail)
        prev_len = len(res)
        for fact in new_facts:
            if fact not in res:
                res[fact] = depth
        depth += 1

    return res


def backward_chain_(u, depth, rules, facts, max_depth, ances):
    INF = 100000000
    if u in facts:
        return INF
    if u in ances or depth == max_depth:
        return depth

    res = depth
    for rule in [x for x in rules if x[1] == u]:
        head, _ = rule
        tmp = INF
        for lit in head:
            ances.add(u)
            tmp = min(tmp, backward_chain_(lit,
                depth + 1, rules, facts, max_depth, ances))
            ances.remove(u)
        res = max(res, tmp)
    return res


def backward_chain(query, rules, facts, max_depth):
    return backward_chain_(query, 0, rules, facts, max_depth, set())


def process_example(example, max_depth):
    [random.shuffle(rule[0]) for rule in example['rules']]
    random.shuffle(example['rules'])
    random.shuffle(example['facts'])

    res = forward_chain(example['rules'], example['facts'])

    example['label'] = 1 if example['query'] in res else 0

    if example['label'] == 0:
        depth = backward_chain(example['query'], example['rules'], example['facts'], max_depth + 1)
    else:
        depth = res[example['query']]

    example['depth'] = depth


def sample_one_example(vocab, min_pred_num, max_pred_num, max_depth, algo):
    pred_num = random.randint(min_pred_num, max_pred_num)
    preds = random.sample(vocab, pred_num)
    if algo == 'RP':
        rules, facts, query = sample_rule_priority(preds)
    if algo == 'LP':
        rules, facts, query = sample_label_priority(preds)
    if algo == 'LP_STAR':
        rules, facts, query = sample_lp_star(preds)

    if query is None:
        return None

    example = {
        'preds': preds,
        'rules': rules,
        'facts': facts,
        'query': query
    }

    process_example(example, max_depth)

    return example


def sample_examples(example_num, vocab, min_pred_num, max_pred_num, max_depth, algo):
    examples = []
    for _ in tqdm(range(0, example_num)):
        example = None
        while example is None:
            example = sample_one_example(vocab, min_pred_num, max_pred_num, max_depth, algo)
        examples.append(example)
    return examples


def stats(examples):
    label_sum = 0.0
    depth_sum = 0.0

    example_num = len(examples)
    if example_num == 0:
        return
    for example in examples:
        label_sum += example['label']
        depth_sum += example['depth']
    print('# of examples:', example_num)
    print('percentage of positive example:', label_sum / example_num)
    print('avg depth:', depth_sum / example_num)


def write_examples(examples, output_file):
    random.shuffle(examples)
    with open(output_file, 'w') as fout:
        json.dump(examples, fout)


def main():
    args = init()
    vocab = read_vocab(args.vocab_file)

    if args.balance_by_depth:
        examples = {}
        example_num = args.example_num

        keys = [x for x in range(0, args.max_depth + 1)]
        for k in keys:
            examples[k] = []

        while True:
            examples_ = sample_examples(1000,
                vocab, args.min_pred_num, args.max_pred_num, args.max_depth, args.algo)
            for example in examples_:
                if example['depth'] > args.max_depth:
                    continue
                key = example['depth']

                if len(examples[key]) < args.example_num:
                    examples[key].append(example)

            if all([len(examples[k]) == args.example_num for k in keys]):
                break

        examples = [x for k in keys for x in examples[k]]

    else:
        examples = sample_examples(args.example_num,
            vocab, args.min_pred_num, args.max_pred_num, args.max_depth, args.algo)

    stats(examples)
    write_examples(examples, args.output_file)


if __name__ == '__main__':
    main()