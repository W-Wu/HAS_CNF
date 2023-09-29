import sys
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(x,y):
    return mean_squared_error(x,y,squared=False)

res_path = sys.argv[1]
res_dic = np.load(res_path,allow_pickle=True).item()
label_path  = 'data/somos-lab.npy'
label_dic = np.load(label_path,allow_pickle=True).item()
NLL_all_dic = np.load(res_path.replace('outcome','NLL_all'),allow_pickle=True).item()
NLL_all = np.mean([v.mean() for v in NLL_all_dic.values()])
NLL_avg_dic = np.load(res_path.replace('outcome','NLL_avg'),allow_pickle=True).item()
NLL_avg = np.mean(list(NLL_avg_dic.values()))

y_true, y_pred=[], []
std, std_lb=[], []

for utt in res_dic:
    y = np.round(np.array(res_dic[utt]))
    y_pred.append(np.mean(y)) 
    y_true.append(np.mean(label_dic[utt]))
    std.extend(np.std(y,axis=0))
    std_lb.append(np.std(label_dic[utt],axis=0))

assert len(y_true) == len(y_pred)

RMSE_y = RMSE(y_true,y_pred)
std_error=abs(np.mean(std)-np.mean(std_lb))
std_RMSE=RMSE(std,std_lb)

print("num_sample\tRMSE_y\tNLL_ref\tNLL_all\tstd_RMSE\tstd_bar\tstd_error")
print(f"{len(res_dic[utt])}\t{RMSE_y}\t{np.array(NLL_avg).mean()}\t{np.array(NLL_all).mean()}\t{std_RMSE}\t{np.mean(std)}\t{std_error}")

