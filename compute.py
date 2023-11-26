from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse
import os
from eval import evaluate
from model.contrast import ContrastModel

import utils

from get_trans_model_new import get_trans_model
from get_recons_model_new import get_recons_model
from train_grad_MIE import MINE_DV, MINE_NWJ, MIGE, infoNCE, MINE_NWJ_sim
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='WebOfScience', choices=['WebOfScience', 'nyt', 'rcv1'], help='Dataset.')
parser.add_argument('--batch', type=int, default=12, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=100, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false', help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')

if __name__ == '__main__':
    args = parser.parse_args()
    #print("???????????????????????????????????????????")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    data_path = os.path.join('data', args.data)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    #label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    #num_class = len(label_dict)
    #model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
    #                                      contrast_loss=args.contrast, graph=args.graph,
    #                                      layer=args.layer, data_path=data_path, multi_label=args.multi,
    #                                      lamb=args.lamb, threshold=args.thre, tau=args.tau)
    MINE=MINE_NWJ()   
    MINE2=MINE_NWJ()                         
    model_gen,tokenizer_gen,optimizer_gen=get_trans_model(args,size1=768,size2=768, num=len(label_dict))
    model_recons,optimizer_recons=get_recons_model(args,size1=768,data_path=data_path)
    #total_params = sum(p.numel() for p in model.parameters())
    #total_params += sum(p.numel() for p in model.buffers())
    total_params += sum(p.numel() for p in model_gen.parameters())
    total_params += sum(p.numel() for p in model_gen.buffers())
    total_params += sum(p.numel() for p in model_recons.parameters())
    total_params += sum(p.numel() for p in model_recons.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params/(1024*1024):.2f}M total parameters.')
    assert 0==1