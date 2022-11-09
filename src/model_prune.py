import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def model_prune(model, prune_ratio):
   parameters_to_prune = []
   for n, m in model.named_modules():
       for name, _ in m.named_parameters():
           if name=='weight' or name=='bias':
               parameters_to_prune.append((m,name))
               #print(n, name)
   prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount = prune_ratio)
   for module, weight in parameters_to_prune:
       prune.remove(module, weight)

def load_model(ckpt):
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    return model

path='tmp/wiki2/gpt2'
path='tmp/squad/t5'
#params = torch.load(path, map_location='cpu')
model = load_model(path)
model_prune(model, 0.1)
