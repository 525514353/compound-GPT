import torch
from torch import nn
criterion = nn.MSELoss()
def caculate(loader,model):
    losses=[]
    model.eval()
    for batch in loader:
        with torch.no_grad():
            labels = batch['logkOH•']
            batch['input_ids']=batch['input_ids'].to(torch.int64)
            logits = model(batch['input_ids'],batch['attention_mask']).squeeze()[torch.arange(0,len(batch['input_ids'])),batch['length']]
            loss = criterion(logits, labels)
            losses.append(loss.unsqueeze(dim=0))
    torch.cuda.empty_cache()
    return torch.sqrt(torch.mean(torch.cat(losses,dim=0))).cpu().numpy()


from sklearn.metrics import r2_score
def caculater2(loader,model):
    y_true=[]
    y_pre=[]
    model.eval()
    for batch in loader:
        with torch.no_grad():
            labels = batch['logkOH•']
            batch['input_ids']=batch['input_ids'].to(torch.int64)
            logits = model(batch['input_ids'],batch['attention_mask']).squeeze()[torch.arange(0,len(batch['input_ids'])),batch['length']]
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(logits.cpu().numpy())
    return r2_score(y_true,y_pre)

import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, Trainer, TrainingArguments,GPT2DoubleHeadsModel,GPT2PreTrainedModel,GPT2ForSequenceClassification
from datasets import load_dataset

# 定义标记器和模型配置
tokenizer = GPT2Tokenizer.from_pretrained(r"D:\system\桌面\lcm-code\tokenizers_lcm\tokenizer_gpt100.json")
tokenizer.pad_token = tokenizer.eos_token


from torch import nn
from torch.nn.utils import prune

class GPT2LinearOutput(nn.Module):
    def __init__(self,n_block,drop,purn):
        super(GPT2LinearOutput, self).__init__()
        self.config=GPT2Config.from_pretrained('D:\system\桌面\lcm-code\pre_training\chem_gpt100')
        self.config.pad_token_id=tokenizer.pad_token_id
        self.config.num_attention_heads=4
        self.config.num_hidden_layers=n_block
        self.config.hidden_size=512
        self.gpt2 = GPT2Model(self.config)  #  使用预训练的GPT2模型
        self.linear = nn.Linear(self.config.hidden_size, 1)  # 输出维度为1的线性层
        for name, module in self.gpt2.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = drop
        for name, module in self.gpt2.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=purn)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids,attention_mask=attention_mask,output_attentions=True)
        last_hidden_states = outputs.last_hidden_state
        linear_output = self.linear(last_hidden_states)  # 只使用最后一个位置的隐藏状态
        return linear_output,outputs['attentions']
