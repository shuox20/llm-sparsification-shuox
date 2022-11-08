import torch
from transformers import AutoModel

def estimate_sparsity(model, lower = -7, upper = 4):
    cnt = torch.zeros(upper-lower+2)
    for n,p in model.named_parameters():
        log_p = torch.log(torch.abs(p))
        cnt_module = torch.histc(log_p, bins = upper-lower, min = lower, max = upper)
        cnt_module = torch.cat((torch.sum(log_p < lower), cnt_module, torch.sum(log_p > upper)))
        cnt += cnt_module
    return cnt

def estimate_sparsity(model, model_name=None, lower = -9, upper = 1):
    cnt = torch.zeros(upper-lower+2)
    for n,p in model.named_parameters():
        log_p = torch.log(torch.abs(p))
        cnt_module = torch.histc(log_p, bins = upper-lower, min = lower, max = upper)
        cnt_module = torch.cat((torch.tensor(torch.sum(log_p < lower)).reshape(1), cnt_module, torch.tensor(torch.sum(log_p > upper)).reshape(1)))
        cnt += cnt_module
    plt.plot(list(range(lower-1, upper+1)), torch.log10(cnt))
    plt.xlabel('log scale')
    plt.ylabel('log10(num of params)')
    plt.savefig(f'{model_name}_total.pdf')
    return cnt

def estimate_sparsity_layer(model, model_name=None, lower = -9, upper = 1):
    cnt_layer = []
    for m in model.encoder.block:
        cnt = 0
        total_cnt = 0
        for n,p in m.named_parameters():
            log_p = torch.log(torch.abs(p))
            cnt += torch.sum(log_p >0)
            total_cnt += log_p.numel()
        cnt_layer.append(cnt/total_cnt)
    
    for m in model.decoder.block:
        cnt = 0
        total_cnt = 0
        for n,p in m.named_parameters():
            log_p = torch.log(torch.abs(p))
            cnt += torch.sum(log_p >0)
            total_cnt += log_p.numel()
        cnt_layer.append(cnt/total_cnt)
    plt.plot(list(range(len(cnt_layer))), torch.log10(torch.tensor(cnt_layer)))
    plt.xlabel('layer')
    plt.ylabel('log10(num of large params)')
    plt.savefig(f'{model_name}_per_layer.pdf')
    return cnt_layer    


model_name = "microsoft/deberta-v2-xxlarge"
model = AutoModel.from_pretrained(model_name, cache_dir = 'cache')
result = estimate_sparsity(model, 'deberta-v2-xxlarge')
print(torch.sum(result[-2:])/torch.sum(result)) # ratio of param > 1
result = estimate_sparsity_layer(model, 'deberta-v2-xxlarge')

model_name = "gp2-xl"
model = AutoModel.from_pretrained(model_name, cache_dir = 'cache')
result = estimate_sparsity(model, 'gpt2-xl')
print(torch.sum(result[-2:])/torch.sum(result)) # ratio of param > 1
result = estimate_sparsity_layer(model, 'gpt2-xl')

model_name = "t5-3b"
model = AutoModel.from_pretrained(model_name, cache_dir = 'cache')
result = estimate_sparsity(model, 't5-3b')
print(torch.sum(result[-2:])/torch.sum(result)) # ratio of param > 1
result = estimate_sparsity_layer(model, 't5-3b')
