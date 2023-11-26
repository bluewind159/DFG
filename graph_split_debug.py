import torch
from transformers import BertConfig
from model.contrast_attention import BertEmbeddings
from model.graph_split_GAT import GraphEncoder
config=BertConfig.from_pretrained('bert-base-uncased')
embeddings=BertEmbeddings(config).cuda()
graph_split = GraphEncoder(config, graph=1, layer=1, data_path='./data/rcv1')
graph_split = graph_split.cuda()
a=torch.randn(6,len(graph_split.label_dict),768).cuda()
label=torch.ones(6,len(graph_split.label_dict)).cuda()
recons_result=graph_split(a, label, lambda x: embeddings(x)[0])
print(recons_result.shape)