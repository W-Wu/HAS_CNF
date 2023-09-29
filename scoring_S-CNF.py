import numpy as np
import sys
import torch
from modules import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from statsmodels.stats import inter_rater
from sklearn.metrics import mean_squared_error

def compute_NLL(predictions,targets):
    predictions= torch.log(predictions)
    xent = torch.sum(predictions*targets,dim=-1)
    return -xent

def RMSE(x,y):
    return mean_squared_error(x,y,squared=False)


maj_path = 'data/msp-lab-maj.npy'
label_path = 'data/msp-lab.npy'
maj_dic = np.load(maj_path,allow_pickle=True).item()
label_dic = np.load(label_path,allow_pickle=True).item()

num_classes=5

res_path = sys.argv[1] 
res_dic = np.load(res_path,allow_pickle=True).item()

NLL_all,NLL_maj = [],[]
y_true, y_pred= [], []
std_lb, std= [], []
kappa_table =[]

label_kappa = 0.254

for utt, v in res_dic.items():
    label = torch.from_numpy(np.array(label_dic[utt])).float()
    num_rater, output_dim_label = label.size()
    maj_id = maj_dic[utt]
    std_lb.append(torch.std(label,dim=0))

    label_avg = torch.mean(label.float(),dim=0)
    v=torch.from_numpy(v).squeeze(0)
        
    num_samples, output_dim = v.size()

    y = F.softmax(v,dim=-1)
    std.append(torch.std(y,dim=0))
    y_bar = torch.mean(y,dim=0)
    kappa_table.append(torch.sum(torch.nn.functional.one_hot(torch.argmax(y,dim=-1),num_classes=num_classes),dim=0))

    assert output_dim == output_dim_label

    loss_all = compute_NLL(predictions=y_bar,targets=label_avg)
    NLL_all.append(loss_all)
    if maj_id!=-1:
        loss_maj = compute_NLL(predictions=y_bar,targets=torch.nn.functional.one_hot(torch.argmax(label_avg),num_classes=num_classes))
        NLL_maj.append(loss_maj)

    avg_max_id = torch.argmax(y_bar,dim=-1)
    
    if maj_id !=-1:
        y_true.append(maj_id)
        y_pred.append(avg_max_id)

kappa_table = torch.stack(kappa_table, dim=0)
kappa = inter_rater.fleiss_kappa(kappa_table, method='fleiss')

Acc = accuracy_score(y_true, y_pred)
nll_all_mean=sum(NLL_all).item()/len(NLL_all)
nll_maj_mean=sum(NLL_maj).item()/len(NLL_maj)

std=torch.stack(std)
std_lb=torch.stack(std_lb)
std_avg = torch.mean(std)

std_RMSE=RMSE(std,std_lb)
std_error=abs(torch.mean(std)-torch.mean(std_lb))
kappa_error=abs(kappa-label_kappa)

print("N\tAcc\tNLL-maj\tNLL-all\tstd_RMSE\tstd_bar\tstd_error\tkappa\tkappa_error")
print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(num_samples,Acc,nll_maj_mean,nll_all_mean,std_RMSE,std_avg,std_error,kappa,kappa_error))