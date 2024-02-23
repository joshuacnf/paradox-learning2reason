import torch
torch.cuda.empty_cache()#oliver is to be blamed for this
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import json
import random
import argparse
from tqdm import tqdm

from model import LogicBERT

device = 'cpu'

class LogicDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        # [CLS] Query : Alice is C . A . C . A and B , C .
        text = ""
        text += "[CLS] Start Query : "
        text += "Alice is " + example["query"] + " . "

        for fact in example["facts"]:
            text +=  fact
            text += " . "

        for rule in example["rules"]:
            text +=  " and ".join(rule[0])
            text += " , "
            text += rule[-1]
            text += " . "

        return text, example["label"], example["depth"]


    @classmethod
    def initialze_from_file(cls, file):
        if "," in file:
            files = file.split(",")
        else:
            files = [file]
        all_examples = []
        for file_name in files:
            with open(file_name) as f:
                examples = json.load(f)
                for example in examples:
                    if example['depth'] < 6:
                        all_examples.extend([example])
            # with open(file_name) as f:
            #     examples = json.load(f)
            #     all_examples.extend(examples)
        return cls(all_examples)


def init():
    global device

    parser = argparse.ArgumentParser()

    # was already here
    parser.add_argument('--data_file', type=str,)
    parser.add_argument('--vocab_file', type=str, default='vocab.txt')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cuda_core', default='0', type=str)
    
    # oliver added
    parser.add_argument('--dataset_path', default='', type=str)
    parser.add_argument('--dataset', default='', type=str) 
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1.0, type=float)
    parser.add_argument('--max_cluster_size', default=10, type=int)
    parser.add_argument('--log_file', default='log.txt', type=str)
    parser.add_argument('--output_model_file', default='model.pt', type=str)
    
    args = parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r') as fin:
        vocab = [line.strip() for line in fin.readlines()]
    vocab += ['[CLS]', 'Start', 'Query', ':', 'Alice', 'is', '.', 'and', ',']
    vocab = set(vocab)
    print('vocabulary size: ', len(vocab))
    return vocab


def gen_word_embedding(vocab):
    word_emb = {}
    for word in vocab:
        word_emb[word] = torch.cat((torch.randn(59).to(device), torch.zeros(5).to(device)))
        word_emb[word] /= torch.norm(word_emb[word])
    word_emb['.'] = torch.cat((torch.zeros(59).to(device), torch.ones(1).to(device), torch.zeros(4).to(device)))
    word_emb['[CLS]'] = word_emb['.']

    return word_emb


def gen_position_embedding(n):
    P = torch.randn(n, 64).to(device)
    for i in range(0, n):
        P[i] /= torch.norm(P[i])
    position_emb = torch.zeros(n, 768).to(device)
    for i in range(1, n):
        for j, k in enumerate([3, 5, 7, 1, 0, 2, 4, 6]):
            if i - k >= 0:
                position_emb[i, 64*j:64*(j+1)] = P[i-k]

    for j, k in enumerate([6, 4, 4, 6, 0, 4, 7, 7]):
        position_emb[0, 64*j:64*(j+1)] = P[k]

    return position_emb


def tokenize_and_embed(sentence, word_emb, position_emb):
    seq = [token for token in sentence.split(' ') if token != '']
    x = torch.zeros(len(seq), 768).to(device)
    for i, word in enumerate(seq):
        x[i, :] = torch.cat((torch.zeros(64 * 8).to(device), word_emb[word], torch.zeros(64 * 3).to(device))) + position_emb[i]
    return x
    

# based on PGC repo: pgc/train.py
def train_model(model, train, valid, test,
                lr, weight_decay, batch_size, max_epoch,
                log_file, output_model_file, dataset_name, word_emb, position_emb):
    valid_loader, test_loader = None, None
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    if valid is not None:
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
    if test is not None:
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    max_valid_ll = -1.0e7
    model = model.to(device)
    model.train()
    for epoch in range(0, max_epoch):
        print('Epoch: {}'.format(epoch))

        # step in train
        batch_counter = 0
        for x_batch, labels, something_else_lol in train_loader:
            # temporary mini-train
            # if batch_counter > 10:
            #     break

            # gpu 
            #x_batch = x_batch.to(device)

            # forward passes
            y_batch = []
            optimizer.zero_grad()
            for sentence in x_batch:
                input_state = tokenize_and_embed(sentence, word_emb, position_emb)
                m_out = model(input_state)
                y = m_out[0, 255]
                # print(y.item()) # print logit value
                y_batch.append(y)
            y_batch = torch.stack(y_batch, dim=0)
            
            y_batch = y_batch.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            torch.log(y_batch)
            #torch.log(y_batch)

            # print(y_batch)
            # print(labels)

            bce = torch.nn.BCEWithLogitsLoss() #get the BCE loss function
            loss = bce(y_batch, labels) #compute loss

            # ####
            # # temporarily make the params retain their gradients so we can view em
            # for name, param in model.named_parameters():
            #     param.retain_grad()
            # ####

            loss.backward()#back propogate (compute gradients)
            optimizer.step()#update parameters according to this batch's gradients

            ######### debugging prints:  ########################################################################## 
            # print_nonzero_params = False
            # print_gradients = False

            # if print_nonzero_params:
            #     # show (nonzero) parameters (to see if they are like the times (a changin))
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             #print(name, param.data)#just print all the params
            #             nonzero_mask = torch.ne(param.data, 0)
            #             nonzero_entries = torch.masked_select(param.data, nonzero_mask)
            #             print(name, nonzero_entries.data)

            # if print_gradients:
            #     # print gradients but only for the leaf nodes (where they should be stored after backward())
            #     for name, param in model.named_parameters():
            #         if param.requires_grad and param.is_leaf:
            #             print(param.grad)
            ##########################################################################

            print('Epoch {}, Batch Loss: {}'.format(epoch, loss.item()))

            batch_counter += 1

        # compute accuracy on train, valid and test
        print('we are here 1')
        train_acc = evaluate(model, train_loader, word_emb, position_emb)
        print('we are here 2')
        valid_acc = evaluate(model, valid_loader, word_emb, position_emb)
        print('we are here 3')
        test_acc = evaluate(model, test_loader, word_emb, position_emb)
        print('we are here 4')

        print('Epoch {}; train acc: {}; valid acc: {}; test acc: {}'.format(epoch, train_acc, valid_acc, test_acc))

        
        with open(log_file, 'a+') as f:
            f.write('{} {} {} {}\n'.format(epoch, train_acc, valid_acc, test_acc))

        print('we are here 5')

        if output_model_file != '':# and valid_ll > max_valid_ll:
            torch.save(model, output_model_file)
            #max_valid_ll = valid_ll

        print('we are here 6')



def evaluate(model, dataset_loader, word_emb, position_emb):
    accs = []
    dataset_len = 0
    counter = 0
    for x_batch, labels, something_else_lol in dataset_loader:
        # counter += 1
        # if counter > 10:
        #     break
        #x_batch = x_batch.to(device)
        #y_batch = []
        batch_correct = 0
        batch_total = 0
        for sentence,label in zip(x_batch,labels):
            input_state = tokenize_and_embed(sentence, word_emb, position_emb)
            m_out = model(input_state)
            y = m_out[0, 255]
            # print(y.item()) # print logit value
            #y_batch.append(y)
            correct_prediction = ((y>.5) == label)
            accs.append(correct_prediction)
            batch_correct += correct_prediction
            batch_total += 1
        #print('evaluation batch accuracy: {}'.format(batch_correct/batch_total))
    acc = sum(accs) / len(accs)
    return acc

def main():
    args = init()

    #dataset = LogicDataset.initialze_from_file(args.data_file)
    #train, valid, test = dataset.split_dataset()

    train = LogicDataset.initialze_from_file(args.data_file+'_train')
    valid = LogicDataset.initialze_from_file(args.data_file+'_val')
    test = LogicDataset.initialze_from_file(args.data_file+'_test')
    
    vocab = read_vocab(args.vocab_file)
    word_emb = gen_word_embedding(vocab)
    position_emb = gen_position_embedding(1024)

    model = LogicBERT()
    model.to(device)

    #train, valid, test = load_data(args.dataset_path, args.dataset)

    train_model(model, train=train, valid=valid, test=test,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epoch=args.max_epoch,
        log_file=args.log_file, output_model_file=args.output_model_file,
        dataset_name=args.dataset, word_emb=word_emb, position_emb=position_emb)

    """ old code (evaluate.py) that checks the model's correctness
    correct_counter = 0

    for index in tqdm(range(len(val_dataset))):        
        text, label, depth = val_dataset[index]

        # skip examples of depth > 10
        if depth > 10:
            continue

        with torch.no_grad():
            input_states = tokenize_and_embed(text, word_emb, position_emb)
            output = model(input_states)

            if (output[0, 255].item() > 0.5) == label:
                correct_counter += 1                
            else:
                print('Wrong Answer!')
                exit(0)

    print(f'AC: {correct_counter} tests passed')
    """

if __name__ == '__main__':
    main()
