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
    def __init__(self, size1, data_path, embeddings):
        super(recons, self).__init__()
        self.size=size1
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        self.recons=nn.Linear(2*768,768)
        self.act = ACT2FN["gelu"]
        self.embeddings=embeddings
        self.id_embedding = nn.Embedding(len(self.label_dict) + 1, size1, len(self.label_dict))
        label_range = torch.arange(len(self.label_dict))
        self.label_id = label_range
        self.label_id = nn.Parameter(self.label_id, requires_grad=False)
        self.dropout=nn.Dropout(0.1)
        self.use_memory=False
        
    def forward(self, x, labels):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name
        label_emb = self.embeddings(self.label_name)[0]
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        label_emb = label_emb.unsqueeze(0)
        expand_size = label_emb.size(-2) // self.label_name.size(0)
        id_emb = self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1, self.size)
        label_emb=label_emb+id_emb.repeat(label_emb.size(0),1,1)
        label_emb = label_emb*labels.unsqueeze(-1)
        label_emb = self.dropout(label_emb)
        result=self.recons(torch.cat([x,label_emb],dim=-1))
        return result

def get_recons_model(args=None,size1=768,data_path=None,embeddings=None):
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=recons(size1, data_path, embeddings)
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
