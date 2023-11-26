import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.manifold import TSNE
import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
def scatter_sample(src, adj, temperature):
    gumbel = torch.distributions.Gumbel(torch.tensor([0.]).to(src.device), 
            torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src+1e-16)
    logit = (log_prob + gumbel) / temperature
    return softmax(logit, adj)

def uniform_prior(adj):
    deg=adj.sum(-1)
    return 1./(deg.unsqueeze(-1)+1e-16)
    
def get_reparam_num_neurons(out_channels, reparam_mode):
    if reparam_mode is None or reparam_mode == "None":
        return out_channels
    elif reparam_mode == "diag":
        return out_channels * 2
    elif reparam_mode == "full":
        return int((out_channels + 3) * out_channels / 2)
    else:
        raise "reparam_mode {} is not valid!".format(reparam_mode)
        
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid_sample(logits, temperature, sent_ids):
    y = logits + sample_gumbel(logits.size()) - sample_gumbel(logits.size())
    #mask=torch.ones(y.size(1),y.size(1))-torch.eye(y.size(1))
    #mask=mask[sent_ids].cuda()
    return F.sigmoid(y / temperature)
    
def gumble_nei_sample(logits, temperature, sent_ids):
    gumbel = torch.distributions.Gumbel(torch.tensor([0.]).to(src.device), 
            torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src+1e-16)
    logit = (log_prob + gumbel) / temperature
    return Nsoftmax(logit, sent_ids, num_nodes)

#def Nsoftmax(logit, sent_ids, num_nodes=3):

def gumbel_sigmoid(logits, temperature, sent_ids=None, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_sigmoid_sample(logits, temperature, sent_ids)
    latent_dim = logits.size(1)
    categorical_dim = logits.size(0) 
    if not hard:
        return y#.view(-1, latent_dim * categorical_dim)
    else:
        assert 0==1
    '''
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard#.view(-1, latent_dim * categorical_dim)
    '''
def gumbel_softmax_sample(logits, temperature, sent_ids=None):
    y = logits #+ sample_gumbel(logits.size())
    #mask=torch.ones(y.size(1),y.size(1))-torch.eye(y.size(1))
    #mask=mask[sent_ids].cuda()
    return masked_softmax(y / temperature,dim=1)

def gumbel_softmax(logits, temperature, sent_ids=None, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature,sent_ids)
    latent_dim = logits.size(1)
    categorical_dim = logits.size(0)
    if not hard:
        return y#.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard#.view(-1, latent_dim * categorical_dim)     