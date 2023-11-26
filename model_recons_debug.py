import torch
from transformers import BertConfig
from model.contrast import BertModel
from get_recons_model_new import get_recons_model
import os
data_path='./data/rcv1'
print('bert embdding recons!!!!!!!!!!!!!!!!!!!!!!!!!!!')
label_hier = torch.load(os.path.join(data_path, 'slot.pt'))
path_dict = {}
num_class = 0
for s in label_hier:
    for v in label_hier[s]:
        path_dict[v] = s
        if num_class < v:
            num_class = v
num_class += 1
for i in range(num_class):
    if i not in path_dict:
        path_dict[i] = i
label_attention_mask=torch.zeros(num_class,num_class).cuda()
for key, value in path_dict.items():
    label_attention_mask[key,value]=1
    label_attention_mask[value,key]=1
    label_attention_mask[key,key]=1
    label_attention_mask[value,value]=1
print(label_attention_mask.shape)
assert 0==1

config=BertConfig.from_pretrained('bert-base-uncased')
model=BertModel(config)
model=model.cuda()
model_recons,optimizer_recons=get_recons_model(size1=768,data_path='./data/rcv1')
model_recons=model_recons.cuda()
a=torch.randn(6,len(model_recons.label_dict),768).cuda()
label=torch.ones(6,len(model_recons.label_dict)).cuda()
recons_result=model_recons(a,label,model,model)
print(recons_result.shape)
