import torch
from transformers import AutoTokenizer
label_dict = torch.load('bert_value_dict.pt')
label_tree = torch.load('slot.pt')
final_tree = {}
print(label_tree)
print(len(label_tree))
assert 0==1
values=[]
index=torch.arange(len(label_tree))
count=torch.zeros(len(label_tree))
anti_tree={}
for key_parent,value_parent in label_tree.items():
    for value in value_parent:
        anti_tree[value]=key_parent
        
def get_label_degree(label_tree):
    label_degree={}
    origin=[]
    for key_parent,value_parent in label_tree.items():
        flag=True
        for key_child,value_child in label_tree.items():
            if key_parent in value_child:
                flag=False
        if flag:
            origin.append(key_parent)
    label_degree[0]=set(origin)
    count=0
    for i in range(10):
        temp=set()
        for value in label_degree[count]:
            temp=temp.union(label_tree[value])
        if len(temp)>0:
            label_degree[count+1]=temp
        else:
            break
        count+=1
    return label_degree


label_h=get_label_degree(label_tree)
split_h=1 #from [0,1,2,3]
acc=[]
number=0
for uu in range(split_h+1):
    acc+=list(label_h[uu])
    number+=len(label_h[uu])
print(number)
assert 0==1
split_yui=len(label_h)-split_h
new_anti_tree=anti_tree.copy()
for i in range(split_yui):
    values=label_h[len(label_h)-1-i]
    for value in values:
        #print(value)
        parent=value
        for j in range(split_yui-i-1):
            parent=anti_tree[parent]
        if split_yui-i-1>0:
            new_anti_tree[value]=parent

all_features=torch.zeros(len(label_tree))
count=0
print(label_tree)
for i in range(number):
    all_features[acc[i]]=acc[i]
pr
for keys,values in label_h.items():
    if keys>split_h:
        for value in values:
            all_features[value]=new_anti_tree[value]
print(all_features)
assert 0==1
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_dict = {i: tokenizer.decode(v) for i, v in label_dict.items()}
label_name = []
for i in range(len(label_dict)):
    label_name.append(label_dict[i])
label_name = tokenizer(label_name, padding='longest')['input_ids']
label_mask = label_name != tokenizer.pad_token_id