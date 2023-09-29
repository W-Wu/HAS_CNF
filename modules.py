"""
modules for CNFs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import normflows as nf
import numpy as np
from normflows import utils



class TransformerModel(nn.Module):
    def __init__(self,input_dim=768,output_dim=3,output_channel=2,num_pretrain_layers=13,
                d_trans=256, nhead=4, num_encoder_layers=2,d_fc=128,num_fc=0,
                dp = 0.1,device='cuda'):
        super().__init__()
        self.num_pretrain_layers = num_pretrain_layers
        if self.num_pretrain_layers > 1:
            self.layer_weights=nn.Parameter(torch.ones(num_pretrain_layers) /num_pretrain_layers)
        self.fc_embed=nn.Linear(input_dim,d_trans)

        self.pos_encoder = PositionalEncoding(d_trans, dp)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_trans, nhead=nhead, dim_feedforward=d_trans, dropout=dp)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.num_fc=num_fc
        if self.num_fc>=1:
            self.fc = [nn.Linear(d_trans,d_fc)]
            for block_index in range(num_fc-1):
                self.fc.append(nn.Linear(d_fc,d_fc))
                self.fc.append(nn.LeakyReLU())
                self.fc.append(nn.Dropout(dp))
            self.fc = nn.ModuleList(self.fc)
            self.out_params = nn.Linear(d_fc, int(output_dim*output_channel))
        else:
            self.out_params = nn.Linear(d_trans, int(output_dim*output_channel))
        self.output_dim = output_dim
        self.device = device


    def forward(self, src,src_key_padding_mask=None):
        if self.num_pretrain_layers > 1:
            norm_weights=nn.functional.softmax(self.layer_weights, dim=-1)
            src=(src*norm_weights.view(1, 1, -1, 1)).sum(dim=2)

        src = src.permute(1,0,2)
        src = self.fc_embed(src)
        src = F.leaky_relu(src)
        src = self.pos_encoder(src)    
        output = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask)

        # mean pooling
        output = torch.mean(output,dim=0)
        if self.num_fc>=1:
            for fc in self.fc:
                output = fc(output)     
        params = self.out_params(output).split(self.output_dim, dim=1)
        return params
     
class PositionalEncoding(nn.Module):
    #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout= 0.1, max_len = 1800):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2.0) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class linear_enc(nn.Sequential):
    def __init__(
            self,
            input_dim=5,
            output_dim=5,
            activation=torch.nn.LeakyReLU,
            dnn_blocks=2,
            dnn_neurons=15,
            dp=0.1,
            split=2,
        ):
        super().__init__()

        self.append(nn.Linear(input_dim,dnn_neurons))
        self.append(activation())
        self.append(nn.Dropout(p=dp))
        if dnn_blocks>1:
            for block_index in range(dnn_blocks-1):
                self.append(nn.Linear(dnn_neurons,dnn_neurons))
                self.append(activation())
                self.append(nn.Dropout(p=dp))
        self.append(nn.Linear(dnn_neurons,output_dim*split))


class Conditional_DiagGaussian(nf.distributions.base.BaseDistribution):
    def __init__(self, shape, trainable=True,temperature=None):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.temperature = temperature  # Temperature parameter for annealed sampling

    def forward(self, loc,log_scale, num_samples=1):
        if self.temperature:
            log_scale = log_scale+ np.log(self.temperature)

        Flag = 0
        if len(loc.shape) ==3:
            Flag = 1
            batch_size,num_rater,output_dim=loc.shape
            if len(log_scale.shape):    #otherwise decoder tensor single value
                log_scale=log_scale.unsqueeze(2).expand(-1,-1,num_samples,-1).reshape(-1,output_dim)
            loc=loc.unsqueeze(2).expand(-1,-1,num_samples,-1).reshape(-1,output_dim)
        else:
            batch_size, output_dim = loc.shape
            num_rater = 1
            if len(log_scale.shape):    #otherwise decoder tensor single value
                log_scale=log_scale.unsqueeze(1).expand(-1,num_samples,-1).reshape(-1,output_dim)
            loc=loc.unsqueeze(1).expand(-1,num_samples,-1).reshape(-1,output_dim)

        eps = torch.randn(
            (batch_size*num_rater*num_samples,) + self.shape, dtype=loc.dtype, device=loc.device
        )

        z = loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        if Flag ==1:
            return z.reshape(batch_size,num_rater,num_samples,output_dim), log_p.reshape(batch_size,num_rater,num_samples)
        else:
            return z.reshape(batch_size,num_samples,output_dim), log_p.reshape(batch_size,num_samples)


    def log_prob(self, z, loc,log_scale):
        if self.temperature:
            log_scale = log_scale+ np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class Conditional_NormalizingFlow(nn.Module):
    def __init__(self, q0, flows, p=None):
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det -= log_d
        return z, log_det

    def inverse(self, x):
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x, loc,log_scale, reduction=None):
        Flag = 0
        if len(x.shape) ==3:
            Flag = 1
            batch_size,num_samples,output_dim=x.shape
            x = x.reshape(-1,output_dim)
            log_scale=log_scale.unsqueeze(1).expand(-1,num_samples,-1).reshape(-1,output_dim)
            loc=loc.unsqueeze(1).expand(-1,num_samples,-1).reshape(-1,output_dim)
        log_q = torch.zeros(len(x), device=x.device)
        z = x
            
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z,loc,log_scale)

        if reduction=='mean':
            return -torch.mean(log_q)
        else:
            if Flag == 1:
                log_q=log_q.reshape(batch_size,num_samples)
            
            return -log_q

    def reverse_kld(self, loc,log_scale, num_samples=1, beta=1.0, score_fn=True):
        z, log_q_ = self.q0(loc,log_scale,num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_, loc,log_scale)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, loc,log_scale, num_samples=1, alpha=1, dreg=False):
        z, log_q = self.q0(loc,log_scale,num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_, loc,log_scale)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, loc,log_scale, num_samples=1, return_latent=False):
        z, log_q = self.q0(loc,log_scale,num_samples)
        latent = z
        batch_size, num_samples,output_dim = z.shape
        z = z.reshape(-1,output_dim)

        log_q = log_q.reshape(-1,)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if return_latent:
            return z.reshape(batch_size, num_samples,output_dim), log_q.reshape(batch_size, num_samples),latent
        else:
            return z.reshape(batch_size, num_samples,output_dim), log_q.reshape(batch_size, num_samples)

    def log_prob(self, x, loc,log_scale):
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, loc,log_scale)
        return log_q

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
  

def logsoftmax_kv(v,y):
    '''
    v: [B, num_rater, num_sample, output_dim]
    y: [B, num_rater, output_dim]
    '''
    Flag=0
    if len(v.shape)<4:
        v=v.unsqueeze(1)
        y=y.unsqueeze(1)
        Flag=1
    batch_size,num_rater, num_samples,output_dim=v.shape
    log_softmax = nn.LogSoftmax(dim=-1)

    v=v.reshape(-1,output_dim)
    log_softmax_v = log_softmax(v)
    y=y.unsqueeze(2).expand(-1,-1,num_samples,-1).reshape(-1,output_dim)
    v_y = torch.sum(y*log_softmax_v,dim=-1)

    if Flag:
        return v_y.reshape(batch_size,num_samples)
    return v_y.reshape(batch_size,num_rater,num_samples)


class MetricStats_Acc:
    def __init__(self):
        self.clear()

    def clear(self):
        self.correct = 0.0
        self.total = 0.0
        self.ids = []
        self.summary = {}

    def append(self, ids, predictions, targets):
        self.ids.extend(ids)
        remove_nma=targets!= torch.ones(targets.size())*-1
        tmp = torch.logical_and(predictions==targets, remove_nma)
        cor = sum(tmp).item()
        N = sum(remove_nma).item()
        self.correct+=cor
        self.total+=N

    def summarize(self):
        scores = self.correct/self.total
        return scores

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask