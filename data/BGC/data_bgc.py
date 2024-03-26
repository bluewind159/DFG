from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.dom.minidom
import pandas as pd
import json
from collections import defaultdict
from dataloader import multi_label_atomic

np.random.seed(7)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    source = []
    labels = []
    label_dict = {}
    hiera = defaultdict(set)
    with open('bgc.taxonomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: tokenizer.encode(v.lower(), add_special_tokens=False) for v, i in label_dict.items()}
    torch.save(value_dict, 'bert_value_dict.pt')
    torch.save(hiera, 'slot.pt')
    train_data = multi_label_atomic('./BlurbGenreCollection_EN_train.txt')
    eval_data  = multi_label_atomic('./BlurbGenreCollection_EN_dev.txt')
    test_data  = multi_label_atomic('./BlurbGenreCollection_EN_test.txt')
    data=train_data+eval_data+test_data
    for text, cat in tqdm(data):
        text = tokenizer.encode(text.lower(), truncation=True)
        source.append(text)
        labels.append([label_dict[i] for i in cat])
    print(len(labels))


    with open('tok.txt', 'w') as f:
        for s in source:
            f.writelines(' '.join(map(lambda x: str(x), s)) + '\n')
    with open('Y.txt', 'w') as f:
        for s in labels:
            one_hot = [0] * len(label_dict)
            for i in s:
                one_hot[i] = 1
            f.writelines(' '.join(map(lambda x: str(x), one_hot)) + '\n')

    from fairseq.binarizer import Binarizer
    from fairseq.data import indexed_dataset

    for data_path in ['tok', 'Y']:
        offsets = Binarizer.find_offsets(data_path + '.txt', 1)
        ds = indexed_dataset.make_builder(
            data_path + '.bin',
            impl='mmap',
            vocab_size=tokenizer.vocab_size,
        )
        Binarizer.binarize(
            data_path + '.txt', None, lambda t: ds.add_item(t), offset=0, end=offsets[1], already_numberized=True,
            append_eos=False
        )
        ds.finalize(data_path + '.idx')
    train = []
    val=[]
    test = []
    for i in range(len(data)):
        if i<len(train_data):
            train.append(i)
        elif i<len(train_data)+len(eval_data):
            val.append(i)
        elif i<len(train_data)+len(eval_data)+len(test_data):
            test.append(i)
    print(len(train),len(train_data))
    print(len(val),len(eval_data))
    print(len(test),len(test_data))
    torch.save({'train': train, 'val': val, 'test': test}, 'split.pt')

    # inv_label = {i:v for v, i in label_dict.items()}
    # with open('data/rcv1/rcv1_train_all.json', 'w') as f:
    #     for i in train:
    #         line = json.dumps({'token': source[i], 'label': [inv_label[l] for l in labels[i]], 'doc_topic': [], 'doc_keyword': []})
    #         f.write(line + '\n')
    # with open('data/rcv1/rcv1_val_all.json', 'w') as f:
    #     for i in val:
    #         line = json.dumps({'token': source[i], 'label': [inv_label[l] for l in labels[i]], 'doc_topic': [], 'doc_keyword': []})
    #         f.write(line + '\n')
    # with open('data/rcv1/rcv1_test_all.json', 'w') as f:
    #     for i in test:
    #         line = json.dumps({'token': source[i], 'label': [inv_label[l] for l in labels[i]], 'doc_topic': [], 'doc_keyword': []})
    #         f.write(line + '\n')

