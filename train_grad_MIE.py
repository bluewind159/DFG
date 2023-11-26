import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

def infonce_loss(l, m):
    '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)
    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()
    return loss
    
class Net(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.d = d
        self.H = H
        self.fc1 = nn.Linear(self.d, self.H)
        self.fc2 = nn.Linear(self.H, self.H)
        self.fc3 = nn.Linear(self.H, 20)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3
        
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class infoNCE():
    def __init__(self, d=768, num_hidden_unit=768):
        self.model1 = Net(d, num_hidden_unit).cuda()
        self.model2 = Net(d, num_hidden_unit).cuda()
        weight_decay=0.01
        adam_epsilon=1e-6
        learning_rate=1e-4
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model1.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model1.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.model2.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model2.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    def compute_MI(self,xs,ys):
        num_sample=xs.size(-2)
        pred_x = torch.sigmoid(self.model1(xs))
        pred_y = torch.sigmoid(self.model2(ys))
        pred_x = pred_x.transpose(2,1)
        pred_y = pred_y.transpose(2,1)
        loss = infonce_loss(pred_y, pred_x)
        return loss
        
    def train_step(self,xs,ys,scheduler=None, train=True):
        MI = self.compute_MI(xs,ys)
        loss = MI.mean()
        self.model1.zero_grad()
        self.model2.zero_grad()
        loss.backward()
        if train:
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
         
class MINE_DV():
    def __init__(self, d=768, num_hidden_unit=1024):
        self.model1 = Net(d, num_hidden_unit).cuda()
        self.model2 = Net(d, num_hidden_unit).cuda()
        self.ema1 = EMA(self.model1, 0.999)
        self.ema2 = EMA(self.model2, 0.999)
        self.ema1.register()
        self.ema2.register()
        self.optimizer = torch.optim.Adam([
            {'params': self.model1.parameters()},
            {'params': self.model2.parameters()}
            ], lr=1e-4)

    def compute_MI(self,xs,ys):
        num_sample=xs.size(-2)
        pred_x = torch.sigmoid(self.model1(xs))
        pred_y = torch.sigmoid(self.model2(ys))
        us = pred_x @ pred_y.T
        mask = torch.eye(num_sample).cuda()
        n_mask = 1 - mask
        #E_pos = torch.sum(mask * us) / (num_sample)
        #E_neg = torch.log(torch.sum(torch.exp(us) * n_mask + 1e-10) / (num_sample ** 2 - num_sample))
        E_pos = torch.sum(us[(mask>0)]) / (num_sample)
        E_neg = torch.log(torch.sum(torch.exp(us[(n_mask>0)])) / (num_sample ** 2 - num_sample))
        MI = E_pos - E_neg
        return - MI
    
    def train_step(self,xs,ys):
        MI = self.compute_MI(xs,ys)
        loss = MI
        self.model1.zero_grad()
        self.model2.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema1.update()
        self.ema2.update()
        self.ema1.apply_shadow()
        self.ema2.apply_shadow()
    
class MINE_NWJ():
    def __init__(self, d=768, num_hidden_unit=1024):
        self.model1 = Net(d, num_hidden_unit).cuda()
        self.model2 = Net(d, num_hidden_unit).cuda()
        self.optimizer = torch.optim.Adam([
            {'params': self.model1.parameters()},
            {'params': self.model2.parameters()}
            ], lr=1e-4)
    def compute_MI(self,xs,ys):
        num_sample=xs.size(-2)
        pred_x = torch.sigmoid(self.model1(xs))
        pred_y = torch.sigmoid(self.model2(ys))
        us = pred_x @ pred_y.T

        mask = torch.eye(num_sample).cuda()
        n_mask = 1 - mask

        E_pos = torch.sum(us[(mask>0)]) / num_sample
        E_neg = torch.sum((torch.exp(us[n_mask>0] - 1.0))) / (num_sample ** 2 - num_sample)
        MI = E_pos - E_neg
        return - MI
        
    def train_step(self,xs,ys,scheduler=None,train=True):
        MI = self.compute_MI(xs,ys)
        loss = MI
        self.model1.zero_grad()
        self.model2.zero_grad()
        loss.backward()
        if train:
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

class MINE_NWJ_sim():
    def __init__(self, d=768, num_hidden_unit=1024):
        self.model1 = Net(d*2, num_hidden_unit).cuda()
        self.optimizer = torch.optim.Adam([
            {'params': self.model1.parameters()},
            #{'params': self.model2.parameters()},
            #{'params': self.model3.parameters()}
            ], lr=1e-4)
    def compute_MI(self,xs,ys):
        num_sample=xs.size(-2)
        input_pos=torch.cat([xs,ys],dim=-1)
        #ram_idx=torch.randprem(ys.size(0))
        idx=torch.randint(0,ys.size(0)-1,(1,1))
        ys_hat=torch.roll(ys,idx[0][0].item(),dims=0)
        input_neg=torch.cat([xs,ys_hat],dim=-1)
        pos = torch.sigmoid(self.model1(input_pos))
        neg = torch.sigmoid(self.model1(input_neg))
        E_pos = torch.sum(pos) / num_sample*20
        E_neg = torch.sum((torch.exp(neg - 1.0))) / num_sample*20
        MI = E_pos - E_neg
        return - MI
        
    def train_step(self,xs,ys,scheduler=None,train=True):
        MI = self.compute_MI(xs,ys)
        loss = MI
        self.model1.zero_grad()
        #self.model2.zero_grad()
        loss.backward()
        if train:
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
class ScoreEstimator:
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return torch.exp(
            -torch.sum(torch.mul((x1 - x2), (x1 - x2)), dim=-1) / (2 * torch.mul(kernel_width, kernel_width))
        )

    def gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width[..., None] ** 2)
        G_expand = torch.unsqueeze(G, -1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        n_samples = x_samples.size()[-2]
        n_basis = x_basis.size()[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.sqrt(
            torch.sum(torch.mul(x_samples_expand - x_basis_expand, x_samples_expand - x_basis_expand), dim=-1)
        )
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
        kernel_width = torch.reshape(top_k_values[:, -1], x_samples.size()[:-2])
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()
        
class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super().__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        M = torch.tensor(samples.size()[-2]).to(samples.device)
        Kxq = self.gram(x, samples, kernel_width)
        ret = torch.sqrt(M.float()) * torch.matmul(Kxq, eigen_vectors)
        ret *= 1. / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.size()[-2]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)

        eigen_values, eigen_vectors = torch.symeig(Kq, eigenvectors=True, upper=True)

        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(
                torch.reshape(eigen_values, [-1, M]), dim=0)

            eigen_arr = torch.flip(eigen_arr, [-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            eigen_lt = torch.lt(eigen_cum, self._n_eigen_threshold)
            self._n_eigen = torch.sum(eigen_lt)
        if self._n_eigen is not None:
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        eigen_ext = self.nystrom_ext(samples, x, eigen_vectors, eigen_values, kernel_width)
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        M = torch.tensor(M).to(samples.device)
        beta = -torch.sqrt(M.float()) * torch.matmul(torch.transpose(eigen_vectors, -1, -2),
                                                     grad_K1_avg) / torch.unsqueeze(eigen_values, -1)
        grads = torch.matmul(eigen_ext, beta)
        self._n_eigen = None
        return grads
        
def entropy_surrogate(estimator, samples):
    dlog_q = estimator.compute_gradients(samples.detach(), None)
    surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
    return surrogate_cost

class MIGE():
    def __init__(self, d=768, num_samples=1024, threshold=None, n_eigen=None):
        self.spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
        self.spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
    def compute_MI(self, xs, ys):
        xs_ys=torch.cat([xs,ys],dim=-1)
        ans = entropy_surrogate(self.spectral_j, xs_ys) \
              - entropy_surrogate(self.spectral_m, ys)
        return ans