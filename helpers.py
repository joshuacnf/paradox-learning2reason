import torch
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
from collections import defaultdict
class TrainingMeter():
    def __init__(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)

    def update(self, loss_dict):
        for key, item in loss_dict.items():
            self.counter_dict[key] += 1
            self.true_dict[key] += item

    def report(self):
        keys = list(self.counter_dict.keys())
        keys.sort()
        for key in keys:
            print("  {} : {:.7}".format(key, self.true_dict[key] / self.counter_dict[key]))
    
    def clean(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)
def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Full loading failed!! Try partial loading!!")

    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped: " + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)
    print("\n\n")

from torch import nn
def expand_position_embeddings(model, length = None, model_type = "bert"):
    if "bert" in model_type:
        embedding_model = model.bert.embeddings
    original_embedding = embedding_model.position_embeddings.weight.data
    new_embedding = nn.Embedding(length - 500, 1024 if "large" in model_type else 768)
    _init_weights(new_embedding, model.config)
    new_embedding = torch.cat( (original_embedding, new_embedding.weight.data), dim = 0)
    embedding_model.position_embeddings.weight = torch.nn.Parameter(new_embedding)

    embedding_model.register_buffer("position_ids", torch.arange(3000).expand((1, -1)))
    
def _init_weights(module, config):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()