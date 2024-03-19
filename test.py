from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train import BertDataset
from eval import evaluate
from model.contrast_test import ContrastModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}_split.pt'.format(args.extra)),
                            map_location='cpu')
    checkpoint_gen = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}_gen_split.pt'.format(args.extra)),
                            map_location='cpu')
    print('load model from',os.path.join('checkpoints', args.name, 'checkpoint_best{}_split.pt'.format(args.extra)))
    batch_size = args.batch
    device = args.device
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join('data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = False
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn)
    dev = Subset(dataset, split['val'])
    dev = DataLoader(dev, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn)
    model.load_state_dict(checkpoint['param'])

    model.to(device)
    from get_trans_model_new import get_trans_model
    from get_recons_model_normal import get_recons_model
    from train_grad_MIE import MINE_DV, MINE_NWJ, MIGE, infoNCE, MINE_NWJ_sim
    model_gen,tokenizer_gen,optimizer_gen=get_trans_model(args,size1=768,size2=768, num=len(label_dict))
    model_recons,optimizer_recons=get_recons_model(args,size1=768,data_path=data_path)
    model_gen.load_state_dict(checkpoint_gen['param'])
    model_gen.to(device)
    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    model.eval()
    model_gen.eval()
    model_recons.eval()
    pbar = tqdm(test)
    with torch.no_grad():
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True, adv_trans=model_gen, recons=model_recons)
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)
            for l in output['logits']:
                pred.append(torch.sigmoid(l).tolist())

    pbar.close()
    scores = evaluate(pred, truth, label_dict, threshold=(20+threshold)/100)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('macro:', macro_f1, 'micro:', micro_f1)
