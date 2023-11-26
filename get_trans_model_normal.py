import torch
import torch.nn as nn
from torch.nn.functional import softplus
from transformers import AdamW
import numpy as np
import torch.nn.functional as F
from biaffine import DirectBiaffineScorer, DeepBiaffineScorer
import os
import random
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
    
class model_trans(nn.Module):
    def __init__(self, size1, size2):
        super(model_trans, self).__init__()
        self.pos_trans = nn.Linear(size1, size2)
        self.neg_trans = nn.Linear(size1, size2)
        self.act = ACT2FN["gelu"]
        self.dropout=nn.Dropout(0.1)
        self.use_memory=False
    def forward(self, x1):
        pos_result=self.act(self.pos_trans(x1))
        pos_result=self.dropout(pos_result)
        neg_result=self.act(self.neg_trans(x1))
        neg_result=self.dropout(neg_result)
        return pos_result, neg_result

class model_conv(nn.Module):
     def __init__(self, size1):
        super(model_conv, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=5,
                       stride=1,padding=2, dilation=1, groups=1,
                       bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(50),
            nn.Tanh(),
            #nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5,
            #           stride=1,padding=2, dilation=1, groups=1,
            #           bias=True, padding_mode='zeros'),
            #nn.BatchNorm1d(50),
            #nn.Tanh(),
            #nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5,
            #           stride=1,padding=2, dilation=1, groups=1,
            #           bias=True, padding_mode='zeros'),
            #nn.BatchNorm1d(50),
            nn.Linear(768,768),
            nn.Tanh()
        )
        self.use_memory=False
     def forward(self, x1):
        result=self.trans(x1)
        return result
        
class model_gaussian(nn.Module):
    def __init__(self, size1):
        super(model_gaussian, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(size1, size1),
            nn.Tanh(),
            nn.Linear(size1, size1),
            nn.Tanh(),
            nn.Linear(size1, size1),
            nn.Tanh()
        )
        self.use_memory=False
    def forward(self, x1):
        self.gaussian=torch.randn(x1.size()).cuda()
        result=self.trans(self.gaussian)
        return result
             
def get_trans_model(args=None,size1=768,size2=768,num=1):
    tokenizer=None
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    print('ReLU activate')
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=nn.ModuleList()
    for i in range(num):
        model.append(model_trans(size1,size2))
    model.append(nn.LSTM(size1, size1, 1,batch_first=True, bidirectional=True))
    model.append(DeepBiaffineScorer(size1, size1, 1024, 1, hidden_func=F.relu, dropout=0.1,
                 pairwise=False))
    model.append(nn.Linear(in_features=size1, out_features=num))
    #print('use 50 MLP @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #checkpoint_path = '../checkpoints/bert/distangle/model_inf'
    #print('load inf model from', checkpoint_path)
    #checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #model.load_state_dict(checkpoint_dict)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return model.cuda(),tokenizer,optimizer


if __name__ == '__main__':
    main()
