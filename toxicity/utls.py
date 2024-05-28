import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, Trainer, TrainingArguments,GPT2DoubleHeadsModel,GPT2PreTrainedModel,GPT2ForSequenceClassification
from datasets import load_dataset
item='Ames Mutagenicity'
# 定义标记器和模型配置
tokenizer = GPT2Tokenizer.from_pretrained(r"D:\system\桌面\lcm-code\tokenizers_lcm\tokenizer_gpt100.json")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id


from torch import nn
from transformers import AutoModelForSequenceClassification,GPT2ForSequenceClassification,GPT2Config
from torch.nn.utils import prune
class GPT2cls(nn.Module):
    def __init__(self,drop,purn,n_block):
        super(GPT2cls, self).__init__()
        self.config = GPT2Config.from_pretrained('D:\system\桌面\lcm-code\pre_training\chem_gpt100')  #  使用预训练的GPT2模型
        self.config.pad_token_id=tokenizer.pad_token_id
        self.config.num_labels=2
        self.config.num_attention_heads=4
        self.config.num_hidden_layers=n_block
        self.config.hidden_size=512
        self.model = AutoModelForSequenceClassification.from_config(self.config)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = drop
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=purn)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        return outputs

from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, auc,roc_auc_score
from rdkit.Chem import AllChem
def acc_1batch(loader,model):
    predicts=[]
    labels=[]
    model.eval()
    for batch in loader:
        with torch.no_grad():
            label = batch[item].to(torch.int64)
            inputs=torch.cat(batch['input_ids']).unsqueeze(dim=0)
            attention=torch.cat(batch['attention_mask']).unsqueeze(dim=0)
            # batch['input_ids']=batch['input_ids'].to(torch.int64)
            logits = model(input_ids=inputs,attention_mask=attention).logits

            predict=torch.argmax(logits,dim=1)
            predicts.append(predict.flatten().cpu().numpy())
            labels.append(label.flatten().cpu().numpy())

    predicts=np.concatenate(predicts)
    labels=np.concatenate(labels)
    torch.cuda.empty_cache()
    return (predicts==labels).sum()/len(labels)


def auc_score_1batch(loader, model):
    predicts = []
    labels = []
    model.eval()

    for batch in loader:
        with torch.no_grad():
            label = batch[item].to(torch.int64)
            inputs=torch.cat(batch['input_ids']).unsqueeze(dim=0)
            attention=torch.cat(batch['attention_mask']).unsqueeze(dim=0)
            logits = model(input_ids=inputs, attention_mask=attention).logits

            # 获取预测概率
            probabilities = torch.softmax(logits, dim=-1)
            # 获取属于第一个类别的概率，这里假设第一个类别是正类别
            positive_probs = probabilities[:, 1]

            predicts.append(positive_probs.cpu().numpy())
            labels.append(label.cpu().numpy())

    # 重组数据以计算ROC曲线和AUC
    predicts = np.concatenate(predicts)
    labels = np.concatenate(labels)
    torch.cuda.empty_cache()

    return roc_auc_score(labels,predicts)