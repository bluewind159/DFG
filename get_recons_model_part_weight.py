import torch
import torch.nn as nn
from torch.nn.functional import softplus
from transformers import AdamW
import numpy as np
import torch.nn.functional as F
from biaffine import DirectBiaffineScorer, DeepBiaffineScorer
import os
import random
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class recons(nn.Module):
    def __init__(self, size1, data_path):
        super(recons, self).__init__()
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        self.recons=nn.Linear(2*768,768)
        self.compute=nn.Linear(len(self.label_dict),1)
        self.act = nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.use_memory=False
    def forward(self, x, labels, embeddings):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name
        label_emb = embeddings(self.label_name)
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        label_emb = label_emb.unsqueeze(0)*labels.unsqueeze(-1)
        label_emb = self.compute(label_emb.transpose(2,1)).transpose(2,1)
        result=self.recons(torch.cat([x,label_emb],dim=-1))
        return result

def get_recons_model(args=None,size1=768,data_path=None):
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=recons(size1, data_path)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return model.cuda(),optimizer


if __name__ == '__main__':
    main()
