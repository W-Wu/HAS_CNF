import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch


# read
def read_json(json_file):
    f=open(json_file,'r')
    text=f.read()
    dic=json.loads(text)
    return dic

# write
def dump_json(json_dict,json_file):
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

def pick_majority(x):
    cnt = sum(np.isclose(x,max(x)))
    if cnt>1:
        return -1
    else:
        return np.argmax(x)


#### Prepare label and text inputs
mapping={ "normal":[1,0,0], "offensive":[0,1,0],"hatespeech":[0,0,1]}

dic = read_json('/HateXplain/dataset.json')
lab_dic = {}
text_dic = {}
maj_dic = {}

for i,x in dic.items():
    lab_dic[i]=[]
    for anno in x['annotators']:
        lab_dic[i].append(mapping[anno['label']])
    maj_dic[i]=pick_majority(np.mean(lab_dic[i],axis=0))
    text_dic[i]=" ".join(x['post_tokens'])

output_dir='../data/'
np.save(f'{output_dir}/HateXplain-lab.npy',lab_dic)
np.save(f'{output_dir}/HateXplain-lab-maj.npy',maj_dic)


#### Extract BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

dic={}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
for i, wrd in text_dic.items():
    inputs = tokenizer(wrd, return_tensors="pt").to(device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    dic[i]= last_hidden_states.detach().cpu().numpy()   
    
np.save('data/HateXplain-bert-base.npy',dic)

#### Prepare train/val/text split, including cases where all three annotators chose a different class. 
split_json='HateXplain/post_id_divisions.json'
split_dic=read_json(split_json)
split_utts=[]
for split, snts in split_dic.items():
    print(split, len(snts))
    split_utts.extend(snts)

no_maj=[]
for utt in text_dic.keys():
    if utt not in split_utts:
        no_maj.append(utt)

no_maj_valid=[x for i,x in enumerate(no_maj) if i % 10 ==0]
no_maj_test=[x for i,x in enumerate(no_maj) if i % 10 ==1]
no_maj_train=[x for i,x in enumerate(no_maj) if x not in no_maj_valid and x not in no_maj_test]

no_maj_dic={'train': no_maj_train, 'val': no_maj_valid, 'test': no_maj_test}

for split, snts in split_dic.items():
    dic={}
    for x in snts:
        dic[x]={"utt_name":x,
            "token":text_dic[x]}
    for x in no_maj_dic[split]:
        dic[x]={"utt_name":x,
            "token":text_dic[x]}   
    dump_json(dic, f'hx-{split}.json')
