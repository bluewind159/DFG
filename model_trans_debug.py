from get_trans_model_new import get_trans_model
import torch
from transformers import BertConfig
from model.contrast_attention import BertEmbeddings
import os
data_path='./data/rcv1'
label_hier = torch.load(os.path.join(data_path, 'slot.pt'))
path_dict = {}
num_class = 0
for s in label_hier:
    for v in label_hier[s]:
        path_dict[v] = s
        if num_class < v:
            num_class = v
num_class=num_class+1
adv_trans,tokenizer_gen,optimizer_gen=get_trans_model(size1=768,size2=768, num=num_class)
pooled_output=torch.randn(6,768).cuda()
disen_loss=0
flag=torch.zeros(num_class)
pos_temp=[]#torch.zeros(pooled_output.size(0),self.num_class,pooled_output.size(-1)).cuda()
neg_temp=[]#torch.zeros(pooled_output.size(0),self.num_class,pooled_output.size(-1)).cuda()
recons_target=[]
for key in range(num_class):
    try:
        value=self.path_dict[key]
        pos_qwer,neg_qwer=adv_trans[key](pos_temp[value].squeeze())
        recons_target.append(pos_temp[value])
        print('Children Nodes!!!!!!')
    except:
        pos_qwer,neg_qwer=adv_trans[key](pooled_output)
        recons_target.append(pooled_output.unsqueeze(1))
        print('Root Nodes !!!!!!')
    pos_temp.append(pos_qwer.unsqueeze(1))
    neg_temp.append(neg_qwer.unsqueeze(1))
    #flag[key]+=1
pos_temp=torch.cat(pos_temp,dim=1)
neg_temp=torch.cat(neg_temp,dim=1)
assert 0==1
print(flag)
print(pos_temp.shape)
print(neg_temp.shape)
print(pos_temp[0].sum(1))
print(neg_temp[0].sum(1))