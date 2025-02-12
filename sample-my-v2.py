"""
Sample from a trained model
"""
import os
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' 或者指定预训练的 GPT-2 模型 (例如 'gpt2-xl')
checkpoint_name = 'ckpt_batch_32.pt'
out_dir = 'out'  # 如果 init_from 是 'resume'，这里会用到
start = "\n"  # 或 "<|endoftext|>" 或其他自定义起始文本
num_samples = 10  # 生成的样本数
max_new_tokens = 256  # 每个样本生成的最大token数量
temperature = 0.8  # 控制生成随机性的温度
top_k = 200  # top_k 策略
seed = 1337
device = 'cuda:3'  # 设备设置
compile = False  # 是否使用 PyTorch 2.0 编译加速
# 读取输入文件并进行编码
input_file = '/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test.jsonl'
output_file = '/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test_output_gpt2_batch32_256.json'
batch_size = 16
# -----------------------------------------------------------------------------


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
torch.backends.cudnn.allow_tf32 = True  # 允许cudnn加速
device_type = 'cuda' if 'cuda' in device else 'cpu'  # 设备类型
device = torch.device(device)

# model
if init_from == 'resume':
    # 从之前保存的模型恢复
    ckpt_path = os.path.join(out_dir, checkpoint_name)  # 假设保存的检查点路径是 out_dir/ckpt.pt
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint['model_name']
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.load_state_dict(checkpoint['model'])  # 加载模型的状态字典
elif init_from.startswith('gpt2'):
    # 从预训练的 GPT2 模型加载
    model = GPT2LMHeadModel.from_pretrained(init_from)
    tokenizer = GPT2Tokenizer.from_pretrained(init_from)
    
model.eval()
model = model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(res, outfile):
    with open(outfile, 'w', encoding='utf-8') as f:
        for d in res:
            f.writelines(json.dumps(d, ensure_ascii=False))
            f.writelines('\n')

def process_batches(input_file, output_file, model, tokenizer, device, batch_size=4, max_new_tokens=500, temperature=1.0, top_k=40):
    inputs_data = load_jsonl(input_file)
    eos_str = tokenizer.decode([tokenizer.eos_token_id])  # EOS token
    inputs = [d['question']+"\n" for d in inputs_data]
    outputs = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_tokens = [torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device) for text in batch]
        batch_tokens = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long, device=device) for t in batch_tokens]
        x = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True, padding_value=tokenizer.eos_token_id)

        # 生成 attention mask 来标识填充部分
        attention_mask = (x != tokenizer.eos_token_id).long()  # 填充部分的 attention_mask 设为 0
        
        # 使用GPT2LMHeadModel进行生成
        with torch.no_grad():
            outputs_batch = model.generate(
                input_ids=x,
                attention_mask=attention_mask,  # 将 attention_mask 加入
                max_length=x.shape[1] + max_new_tokens,  # 生成的最大长度
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,  # 使用 eos_token_id 作为 pad_token_id
                eos_token_id=tokenizer.eos_token_id
            )
            
            for idx, output in enumerate(outputs_batch):
                output_text = tokenizer.decode(output, skip_special_tokens=True)
                output_text = output_text[len(batch[idx]):]  # 去除输入文本部分
                output_text = output_text.split(eos_str)[0]  # 以EOS符号截断
                outputs.append(output_text)

    write_jsonl(outputs, output_file)
    print(f"Processed outputs saved to {output_file}")

process_batches(input_file, output_file, model, tokenizer, device, batch_size, max_new_tokens, temperature, top_k)