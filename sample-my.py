"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import json
from model import GPTConfig, GPT
from transformers import GPT2Tokenizer

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:4' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(res, outfile):
    f = open(outfile, 'w', encoding='utf-8')
    for d in res:
        f.writelines(json.dumps(d, ensure_ascii=False))
        f.writelines('\n')
    f.close()

# 读取输入文件并进行编码
input_file = '/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test.jsonl'
output_file = '/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test_output.json'
batch_size = 4
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def process_batches(input_file, output_file, model, tokenizer, device, batch_size=4, max_new_tokens=500, temperature=1.0, top_k=40):
    inputs_data = load_jsonl(input_file)
    eos_str = tokenizer.decode([tokenizer.eos_token_id]) # 输出截断符
    inputs = []
    for d in inputs_data:
        inputs.append(d['question'])
    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_tokens = [torch.tensor(tokenizer(text)['input_ids'], dtype=torch.long).to(device) for text in batch]
        batch_tokens = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long, device=device) for t in batch_tokens]
        x = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True, padding_value=0)
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            batch_outputs = []
            for i in range(len(y)):
                output = tokenizer.decode(y[i])
                output = output.split(batch[i])[1]  # 去除input文本
                output = output.split(eos_str)[0]
                batch_outputs.append(output)
            outputs.extend(batch_outputs)
        # print(outputs)
        # exit()
    
    write_jsonl(outputs, output_file)
    print(f"Processed outputs saved to {output_file}")

process_batches(input_file, output_file, model, tokenizer, device, batch_size, max_new_tokens, temperature, top_k)