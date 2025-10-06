import torch
import math
from torch import nn
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
from transformer_architecture import transformer_lm
from pretokenization_example import tokenizer

# cross entropy
# note that, transformer holds all seq related probability since input always batch * n * d, n is seq_len, with masked attention, all time steps are handled at once
# output is batch * n * vocab_size, the second dimension includes softmax info.
# simply, assume batch is 1, seq{x1, x2, x3... xn} is the input. 
# then attention[0][0], a vocab_size vector, gives the softmax logits of x2,
#      attention[0][1] gives the softmax logits of x3, we can use this result to calc cross entropy
# considering first step, watch attention[0][0], let P = o(x2)/sum(exp(x)),
# cross entropy = -logP, when o(x2) is bigger, P approaches more to 1, -logP approches 0, means the predict is accurate.
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    max = inputs.max(dim = -1, keepdim=True)[0]   #[batch... 1]
    target = inputs.gather(-1, targets.unsqueeze(-1)) #[batch ... 1]
    exp_sum_log = torch.exp(inputs - max).sum(-1, keepdim=True).log() #[batch ... 1]
    return torch.mean(max - target + exp_sum_log)   # all these 3 parts are from inputs, so shapes are same, call torch.means

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "lamb": weight_decay,
                    "beta1": betas[0],
                    "beta2": betas[1],
                    "eps": eps}
        super().__init__(params, defaults)
    
    # pass scheduled lr if cosine learning rate decay schedule is used
    def step(self, closure = None, lr = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if lr is None:
                lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lamb = group["lamb"]
            for p in group["params"]:   #group["params"] is a list of tensor, in fact they should be weights in the model
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p, device=p.device))
                v = state.get("v", torch.zeros_like(p, device=p.device)) # get or initial m,v as mentioned in algorithm
                grad = p.grad.data
                m = beta1 * m + (1-beta1)*grad
                v = beta2 * v + (1-beta2)*(grad**2)
                rate = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                p.data = p.data - rate * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * lamb * p.data
                # update state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, t_cos):
    if t < t_warm_up:
        return t/t_warm_up * lr_max
    elif t < t_cos:
        return lr_min + (1 + math.cos( (t-t_warm_up)/(t_cos - t_warm_up)*torch.pi ))*(lr_max - lr_min)/2
    else:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm, eps = 1e-6):
    grads = [param.grad for param in parameters if param.grad is not None]
    l2_norm = torch.sqrt(sum((grad ** 2).sum() for grad in grads))
    # better to keep data in Tensor, avoid use .item(), this will pass data to CPU and break anto differentiation
    # l2_norm = math.sqrt(sum((grad ** 2).sum().item() for grad in grads))
    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)

'''
suppose len(dataset) = n, context_length = m, then we can generate n-m+1 seqs with length m
as dataset is indexed in [0, 1, ... n-1], the start index of last seq is (n - 1) - m + 1 = n-m.
so last possible training seq can start with index n-m-1
'''
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    dataset_tensor = torch.from_numpy(dataset)
    # torch.randin(a, b, size) will generat series of int in [a, b)
    start_indices = torch.randint(0, len(dataset)-context_length, (batch_size,))
    input_seqs, target_seqs = [], []
    for index in start_indices:
        input_seq = dataset_tensor[index: index + context_length]
        target_seq = dataset_tensor[index+1: index + context_length + 1]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    input_seqs = torch.stack(input_seqs).to(device)
    target_seqs = torch.stack(target_seqs).to(device)
    return input_seqs.to(torch.long), target_seqs.to(torch.long)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    model_weights = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model_weight"    : model_weights,
        "optimizer_state" : optimizer_state,
        "iteration"       : iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model_weights = checkpoint["model_weight"]
    optimizer_state = checkpoint["optimizer_state"]
    iteration = checkpoint["iteration"]
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state)
    return iteration

def prepare_data_set():
    # split numpy into 20 files since use 20 processes to generate numpy array
    num_parts = 20
    arrays = [np.fromfile(f'../data/training_seq_data/TinyStoriesV2-GPT4-train_bpe{i}.npy', dtype=np.int16) for i in range(num_parts)]
    dataset = np.concatenate(arrays)
    return dataset

### -----------hyper parameter ------------
epoches = 5000
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
theta = 10000
num_layers = 4
num_heads = 16
lr = 1e-3
lr_max = 1e-3
lr_min = 1e-6
t_warm_up = 100
t_cos = epoches
betas=(0.9, 0.999)
weight_decay=0.01
eps=1e-8
batch_size = 32
max_l2_norm = 1.0

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### --------- end of hyper parameter ------
def training_epoch(iteration: int, dataset: npt.NDArray, model: nn.Module, optim: torch.optim.Optimizer, device:torch.device):
    ## 1. get_batch data, input and target
    input_batch, target_batch = data_loading(dataset, batch_size, context_length, device)
    ## 2. y_hat = model(input)
    model_output = model(input_batch)
    ## 3. loss = cross_entropy(y_hat, target)
    loss = cross_entropy(model_output, target_batch)
    ## 4. backward
    loss.backward()
    ## 5. gradient clipping
    gradient_clipping(model.parameters(), max_l2_norm)
    ## 6. optim.step(), with update_lr if use cosine schedule
    lr = learning_rate_schedule(iteration, lr_max, lr_min, t_warm_up, t_cos)
    optim.step(lr=lr)
    ## 7. zero grad
    optim.zero_grad()
    ## 8. return loss
    return loss

def training():
    print("training start!")
    model = transformer_lm(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device)
    optim = AdamW(model.parameters(), lr, weight_decay, betas, eps)
    dataset = prepare_data_set()
    #for i in range(epoches):
    for i in range(epoches):
        loss = training_epoch(i, dataset, model, optim, device)
        print(f'loss in epoch {i} : {loss}')
    save_checkpoint(model, optim, epoches, f'../data/model_result/TinyStoriesV2-GPT4-train_bpe{i}.model')
    return model

def load_model():
    print('loading model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformer_lm(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device)
    optim = AdamW(model.parameters(), lr, weight_decay, betas, eps)
    # TODO don't hard code here
    load_checkpoint(f'../data/model_result/TinyStoriesV2-GPT4-train_bpe{epoches-1}.model', model, optim)
    return model

def generate(model: transformer_lm, prompt: str, token_handler: tokenizer, max_tokens: int, special_tokens, device=None):
    input = token_handler.encode(prompt)
    input = torch.Tensor(input).to(device=device, dtype=torch.long)
    ret = prompt
    for i in range(max_tokens):
        # print(f'try generating token {i}')
        output:torch.Tensor = model(input)
        # print(f'model generating token {i} over')
        token = torch.argmax(output[-1], dim = -1)
        decoded_token = token_handler.decode([token.item()])
        # print(f'token is {token}, decode as {decoded_token}')
        if decoded_token in special_tokens:
            break
        ret = ret + decoded_token
        input = torch.cat((input, token.unsqueeze(0)))
    return ret

def predict():
    model = load_model()
    print('model loaded')

    token_handler = tokenizer()
    vocab_path = '../data/vocab.bin'
    merges_path = '../data/merges.bin'
    special_tokens = ["<|endoftext|>"]
    print('laoding vocab and merges...')
    token_handler.from_files(vocab_path, merges_path, special_tokens)

    prompt = input('input prompt\n')
    while(prompt != 'exit()'):
        max_tokens = int(input('max tokens:'))
        print('start generating...')
        ret = generate(model, prompt, token_handler, max_tokens, special_tokens, device)
        print(f'result: \n')
        print(ret)
        prompt = input('input prompt\n')

if __name__ == '__main__':
    action = str(input('input action, t for training or p for predict\n'))
    if action == 't' or action == 'training':
        training()
    elif action == 'p' or action == 'predict':
        predict()
#model = training()