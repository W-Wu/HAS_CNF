import json
import numpy as np
import re 

# read
def read_json(json_file):
    f=open(json_file,'r')
    text=f.read()
    dic=json.loads(text)
    return dic

def pick_majority(x):
    cnt = sum(np.isclose(x,max(x)))
    if cnt>1:
        return -1
    else:
        return np.argmax(x)
    
label_path='/MSP-PODCAST-Publish-1.6/label/labels_detailed.json'
label_dic=read_json(label_path)

emo_class={'angry':0, 'disgust':0,'contempt':0,'annoyed': 0,
           'sad':1,'frustrated': 1,'disappointed': 1, 'frustration': 1,'depressed': 1,'concerned': 1, 'concern': 1,'confused': 1,
           'happy':2,'excited': 2, 'amused': 2, 'excitement': 2,
           'neutral':3,
           'other':4
           }

def check_other(x):
    for i,emo in enumerate(x):
        if emo not in emo_class:
            x[i] = 'other'
    x=[*set(x)]
    return x

def extract_bracket(x):
    if "(" in x:
        x = re.search(r'\((.*)\)', x).group(1)
    x = list(map(lambda x: x.lower().strip(), x.split(',')))
    x = list(set(x))
    return x

def construct_distribution(emo_maj):
    dist=np.zeros(5)
    if len(emo_maj)==1:
        primary=emo_maj[0]
        dist[emo_class[primary]]=1 
    else:
        tmp = 1.0/len(emo_maj)
        for emo in emo_maj:
            dist[emo_class[emo]]+=tmp
    assert np.isclose(sum(dist),1.0), dist
    return dist

def process_raw(x):
    x = re.sub("[\(\[].*?[\)\]]", "", x)
    return x.split(',')

dist_dic={}
avg_dist=[]
num_rater=[]
maj_dic={}

for utt,labs in label_dic.items():
    dist_dic[utt]=[]
    for worker,lab in labs.items():
        emo_maj = check_other(extract_bracket(lab['EmoClass_Major']))
        dist = construct_distribution(emo_maj)
        dist_dic[utt].append(dist)
    dist_dic[utt]=np.array(dist_dic[utt])
    avg_dist.append(np.mean(dist_dic[utt],axis=0))
    num_rater.append(len(dist_dic[utt]))
    maj_dic[utt]=pick_majority(np.mean(dist_dic[utt],axis=0))

output_folder = '../data/'
assert len(dist_dic)==len(maj_dic)

np.save(output_folder+'/msp-lab.npy',dist_dic)
np.save(output_folder+'/msp-lab-maj.npy',maj_dic)

    
