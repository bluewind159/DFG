import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch import Tensor, device, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import AdamW, BertConfig, BertLayer, AutoTokenizer
import numpy as np
import torch.nn.functional as F
from biaffine import DirectBiaffineScorer, DeepBiaffineScorer
import os
import random
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
    def __init__(self, size1, data_path):
        super(model_trans, self).__init__()
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        self.dropout=nn.Dropout(0.1)
        self.use_memory=False
        model_name_or_path='bert-base-uncased'
        config = BertConfig.from_pretrained(model_name_or_path)
        config.is_decoder=True
        config.add_cross_attention=True
        self.pos_trans=AttentionTrans()
        self.neg_trans=AttentionTrans()
    def forward(self, encoder_hidden_states, encoder_attention_mask, embeddings):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name
        label_emb = embeddings(self.label_name)
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        label_emb = label_emb.unsqueeze(0).repeat(encoder_hidden_states.size(0),1,1)
        hidden_states = self.dropout(label_emb)
        device = hidden_states.device
        input_shape = encoder_hidden_states.size()[:-1]
        extended_encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask, input_shape, device)
        pos_temp=self.pos_trans(hidden_states, encoder_hidden_states, extended_encoder_attention_mask)
        neg_temp=self.neg_trans(hidden_states, encoder_hidden_states, extended_encoder_attention_mask)
        return pos_temp, neg_temp
        
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
             
class AttentionTrans(nn.Module):
    def __init__(self):
        super(AttentionTrans, self).__init__()
        model_name_or_path='bert-base-uncased'
        config = BertConfig.from_pretrained(model_name_or_path)
        config.is_decoder=True
        config.add_cross_attention=True
        trans=BertLayer(config)
        self.crossattention = trans.crossattention
        self.intermediate = trans.intermediate
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self,hidden_states, encoder_hidden_states, encoder_attention_mask):
        attention_mask = torch.ones(hidden_states.size(0),hidden_states.size(1)).cuda()
        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, device)
        cross_attention_outputs = self.crossattention(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=encoder_attention_mask
                )
        attention_output = cross_attention_outputs[0]
        attention_output = self.intermediate(attention_output)
        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output)
        return attention_output
        
def get_trans_model(args=None,data_path=None,size1=768,size2=768,num=1):
    tokenizer=None
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=model_trans(size1,data_path)
    #for i in range(num):
    #model.append(model_trans(size1,size2))
    #model.append(nn.LSTM(size1, size1, 1,batch_first=True, bidirectional=True))
    #model.append(DeepBiaffineScorer(size1, size1, 1024, 1, hidden_func=F.relu, dropout=0.1,
    #             pairwise=False))
    #model.append(nn.Linear(in_features=size1, out_features=1))
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
