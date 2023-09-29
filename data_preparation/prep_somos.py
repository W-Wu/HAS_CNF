import pandas as pd
import numpy as np
import speechbrain as sb
import json

lab_dic={}
num_rater=[]
WAV_DIC = {}

output_path = '../data/'
audio_root = '/somos/audios_16k/'

for split in ['test','train','valid']:
    path = f'/somos/raw_scores_with_metadata/split1/raw_scores_removed_excess_gt_{split}set.tsv'
    df = pd.read_csv(path, sep='\t', header=0)
    dic_df=df.to_dict('index')
    assert len(df)==len(dic_df)
    dic={}
    for k,v in dic_df.items():
        utt_id = v['utteranceId']
        if utt_id not in dic:
            dic[utt_id] = []
        dic[utt_id].append([v['listenerId'],v['locale'],v['clean'],v['choice']])

    ref_path = f'/somos/training_files/split1/full/{split}_mos_list.txt'
    df = pd.read_csv(ref_path,sep=',',header=0)
    dic_df=df.to_dict('index')
    dic_json={}
    ref_path={}
    for k,v in dic_df.items():
        ref_path[v['utteranceId']]=v['mean']
        fea_path=audio_root+v['utteranceId']
        sig = sb.dataio.dataio.read_audio(fea_path)
        utt_name=v['utteranceId'].split('.')[0]
        dic_json[utt_name]=v
        dic_json[utt_name]['duration']=len(sig)
        WAV_DIC[utt_name]=sig

    for k,v in dic.items():
        lab = [x[-1] for x in v]
        assert np.isclose(ref_path[k+'.wav'],np.mean(lab)),k
        lab_dic[k]=np.array(lab)
        num_rater.append(len(lab_dic[k]))

    assert len(dic_json)==len(dic_df)
    json_file=f"{output_path}/somos-{split}.json"
    with open(json_file, mode="w") as json_f:
        json.dump(dic_json, json_f, indent=2)

np.save(output_path+'somos-wav.npy', WAV_DIC)
np.save(output_path+'/somos-lab.npy',lab_dic)
