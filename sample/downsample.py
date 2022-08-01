import json
import random
from tqdm import tqdm
all_examples = []
for i in tqdm(range(40)):
    file = "DATA/RP_10X/prop_examples_{}.txt.balanced_rulenum".format(i)

    with open(file, 'r') as fin:
        examples = json.load(fin)
    
    all_examples.extend(examples)

total = 7000 * 40
random.shuffle(all_examples)

all_examples = all_examples[:total]
assert(len(all_examples) == total)

file_name = "DATA/RP_10X/prop_examples_all.txt.balanced_rulenum"
train_examples = all_examples[:len(all_examples) // 10 * 8]
dev_examples = all_examples[len(all_examples) // 10 * 8:len(all_examples) // 10 * 9]
test_examples = all_examples[len(all_examples) // 10 * 9:]

with open(file_name + "_train", "w") as f:
    json.dump(train_examples, f)
with open(file_name + "_val", "w") as f:
    json.dump(dev_examples, f)
with open(file_name + "_test", "w") as f:
    json.dump(test_examples, f)
