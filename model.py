import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class StackedGPT2Model(nn.Module):
    def __init__(self, model_name='gpt2', additional_tensor_size=10):
        super(StackedGPT2Model, self).__init__()
        
        # 创建五个 GPT2LMHeadModel 实例
        self.models = nn.ModuleList([GPT2LMHeadModel.from_pretrained(model_name) for _ in range(5)])
        
        # 定义一个定长的张量，形状为 (1, additional_tensor_size)
        self.additional_tensor = nn.Parameter(torch.randn(1, additional_tensor_size), requires_grad=True)

    def forward(self, input_ids, attention_mask=None, targets=None):
        # 在每一层中循环输入
        x = input_ids
        for i, model in enumerate(self.models):
            # 拼接定长张量
            # 先将 additional_tensor 扩展到 (batch_size, additional_tensor_size)
            additional_tensor_expanded = self.additional_tensor.expand(x.size(0), -1)  # (batch_size, additional_tensor_size)
            
            # 将 input_ids 转换为嵌入
            inputs_embeds = model.transformer.wte(x)  # 获取 token embeddings
            
            # 拼接嵌入和定长张量
            combined_inputs = torch.cat((inputs_embeds, additional_tensor_expanded.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)), dim=-1)  # (batch_size, seq_length, n_embd + additional_tensor_size)

            # 通过当前的 GPT2LMHeadModel 进行前向传播
            outputs = model(inputs_embeds=combined_inputs, attention_mask=attention_mask, labels=targets)
            # 获取 logits 和 loss
            logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
            loss = outputs.loss if targets is not None else None
            
            # 使用当前层的 logits 作为下一层的输入
            # 这里我们只取 logits 的最后一个时间步作为下一层的输入
            x = logits[:, -1, :].argmax(dim=-1)  # 取最后一个时间步的预测 token ID
        
        return logits, loss

# 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = StackedGPT2Model()
    
    # 示例输入
    input_ids = torch.randint(0, 50257, (2, 50))  # 假设有一个批次大小为2，序列长度为50的输入
    targets = torch.randint(0, 50257, (2, 50))  # 假设相应的目标

    # 前向传播
    logits, loss = model(input_ids, targets=targets)

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item() if loss is not None else None)