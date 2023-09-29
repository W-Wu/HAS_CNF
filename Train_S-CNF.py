#!/usr/bin/env python3
"""
Code for training S-CNF

> python3 Train_S-CNF.py Train_S-CNF.yaml
"""
import os
import sys
import time
import torch
import random
import ruamel.yaml
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import normflows as nf
from speechbrain.dataio.sampler import BalancingDataSampler
from speechbrain.dataio.dataio import length_to_mask
import logging

from modules import *

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
        
def dataio_prep(hparams):
    label_dic=np.load(hparams['label_path'],allow_pickle=True).item()
    label_dic_maj=np.load(hparams['label_path_maj'],allow_pickle=True).item()
    fea_dic=np.load(hparams['fea_path'],allow_pickle=True).item()
    
    # Define input pipeline
    # For emotion classification
    @sb.utils.data_pipeline.takes("fea_path")
    @sb.utils.data_pipeline.provides("sig","label","maj")
    def data_pipeline(fea_path):
        utt_name=fea_path.split('/')[-1]
        sig = fea_dic[utt_name]
        yield torch.from_numpy(sig).float()
        label = torch.from_numpy(label_dic[utt_name]).float()
        yield label
        maj= label_dic_maj[utt_name]
        yield torch.tensor(maj).long()

    # # For hate speech detection
    # @sb.utils.data_pipeline.takes("utt_name")
    # @sb.utils.data_pipeline.provides("sig","label","maj","duration")
    # def data_pipeline(utt_name):
    #     fea = torch.from_numpy(fea_dic[utt_name]).float()
    #     yield fea.squeeze(0)
    #     yield torch.tensor(label_dic[utt_name]).float()
    #     yield torch.tensor(label_dic_maj[utt_name]).long()
    #     yield len(fea.squeeze(0))

    # Define datasets.
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            dynamic_items=[data_pipeline],
            output_keys=["id", "sig", "label","maj"],
        )
        if dataset == 'train':
            if hparams["sorting"] == "ascending":
                datasets[dataset] = datasets[dataset].filtered_sorted(sort_key="duration")
                hparams["dataloader_options"]["shuffle"] = False

            elif hparams["sorting"] == "descending":
                datasets[dataset] = datasets[dataset].filtered_sorted(sort_key="duration", reverse=True)
                hparams["dataloader_options"]["shuffle"] = False
            elif hparams["sorting"] == "random":
                pass
            else:
                raise NotImplementedError(
                    "sorting must be random, ascending or descending"
                )

    return datasets


# Brain class for conditional softmax flows
class SCNF_Brain(sb.Brain):
    # For emotion classification
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        outputs = self.modules.SSL_encoder(wavs, lens)

        if len(outputs.shape)>3: 
            outputs=outputs.permute(1,2,0,3)    
        else:
            outputs=outputs.transpose(0,1)

        src_key_padding_mask = self.make_masks(outputs,wav_len=lens)
        pred = self.modules.feature_extractor(outputs,src_key_padding_mask=~src_key_padding_mask)
        return pred

    # # For hate speech detection
    # def compute_forward(self, batch, stage):
    #     batch = batch.to(self.device)
    #     feas, lens = batch.sig
    #     pred = self.modules.feature_extractor(feas)
    #     return pred
    
    def make_masks(self, src, wav_len=None, pad_idx=0):
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = length_to_mask(abs_len).bool()
        return src_key_padding_mask
    
    def process_label(self,batch):
        # Padding label to the same number of raters for batch operation. 
        # Padded labels will be masked when computing the loss.
        label = batch.label
        label_maj = batch.maj
        if isinstance(label,sb.dataio.batch.PaddedData):
            label=label.data
            label_maj=label_maj.data
        label_ref = torch.stack([torch.mean(x,dim=0) for x in label])
        num_rater = torch.tensor([len(x) for x in label]).to(self.device)
        label_mask = get_mask_from_lengths(num_rater)
        padded_label = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0.0)
        assert(padded_label.shape[0] == label_mask.shape[0] and padded_label.shape[1] == label_mask.shape[1])
        return label_ref,padded_label,label_mask,label_maj


    def compute_objectives(self, predictions, batch, stage):
        label_ref,label,label_mask,label_maj=self.process_label(batch.to(self.device)) 
        rater_lens=torch.sum(~label_mask,-1)
        # label.shape: [B, num_rater, output_dim]
        # label_ref.shape: [B, output_dim]
        # label_mask.shape: [B, num_rater]
        loc_x, log_scale_x = predictions

        B, num_rater, output_dim = label.size()

        # ELBO = log p(k|v) + log p(v|x) - log q(v|y)
        loc_y, log_scale_y=self.modules.softmax_encoder(label).split(output_dim, dim=-1)

        # log q(v|y), y~q(v|y)
        v,log_prob = self.modules.threshold_base(loc_y,log_scale_y,num_samples=self.hparams.num_elbo)
        log_qvy=torch.mean(log_prob,dim=-1)    
        log_qvy=torch.sum(log_qvy.masked_fill_(label_mask, 0.0),dim=-1)/rater_lens

        # log p(v|x)
        log_pvx = -1.0 * self.modules.flow.forward_kld(v.reshape(B,-1,output_dim), loc_x, log_scale_x)
        log_pvx = torch.mean(log_pvx.reshape(B,num_rater,self.hparams.num_elbo), dim=-1)
        log_pvx=torch.sum(log_pvx.masked_fill_(label_mask, 0.0),dim=-1)/rater_lens

        # log p(k|v)
        log_pkv = logsoftmax_kv(v=v,y=label)
        log_pkv = torch.mean(log_pkv,dim=-1)
        log_pkv=torch.sum(log_pkv.masked_fill_(label_mask, 0.0),dim=-1)/rater_lens

        loss = - (log_pkv+log_pvx-log_qvy)

        # Forward sampling
        v, _ = self.modules.flow.sample(loc_x,log_scale_x, num_samples=self.hparams.num_samples)
        y = F.softmax(v,dim=-1)
        avg_max_id = torch.argmax(torch.mean(y,dim=1),dim=-1)
        self.acc_metric.append(batch.id, targets=label_maj.detach().cpu(), predictions=avg_max_id.detach().cpu())
    
        if stage == sb.Stage.TEST:
            for i in range(len(batch.id)):
                self.test_outcome[batch.id[i]]=v[i].detach().cpu().numpy()

        return loss.mean() 


    def on_stage_start(self, stage, epoch=None):
        self.start_time = time.time()
        self.acc_metric = self.hparams.acc_stats()
        
        if stage == sb.Stage.TEST:
            self.test_outcome = {}
    

    def on_stage_end(self, stage, stage_loss, epoch=None):
        self.elapse_time = time.time() - self.start_time
        stats = {"loss": stage_loss,
                 "Acc": self.acc_metric.summarize(),
                'elapse': self.elapse_time,
                }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stats

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["loss"])

            self.old_lr = old_lr
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": self.old_lr},
                train_stats=self.train_stats,
                valid_stats=stats,
            )

            self.checkpointer.save_and_keep_only(meta=stats, 
                                                 max_keys=["Acc"], 
                                                 num_to_keep=3, 
                                                 name=f"E{self.hparams.epoch_counter.current}")
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            np.save(os.path.join(self.hparams.output_folder,'test_outcome-E{}.npy'.format(self.hparams.epoch_counter.current)),self.test_outcome)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    if '--device' not in sys.argv[1:]:
        run_opts['device']= 'cuda' if torch.cuda.is_available() else 'cpu'

    ruamel_yaml = ruamel.yaml.YAML()
    overrides = ruamel_yaml.load(overrides)
    if overrides:
        overrides.update({'device': run_opts['device']})
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    set_seed(hparams['seed'])

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    

    datasets = dataio_prep(hparams)

    # Define nvp flows
    K = hparams['flow_num_block']
    W = hparams['nvp_hidden_width']
    latent_size = hparams['output_dim']

    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    for i in range(K):
        s = nf.nets.MLP([latent_size, W * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, W* latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]  


    base = Conditional_DiagGaussian(hparams['output_dim'])
    nfm = Conditional_NormalizingFlow(base, flows)
    threshold_base = Conditional_DiagGaussian(hparams['output_dim'])

    hparams["modules"]['flow']=nfm
    hparams["modules"]['threshold_base']=threshold_base
    
    HAS_brain = SCNF_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    HAS_brain.checkpointer.add_recoverable("nfm", nfm)

    train_sampler = BalancingDataSampler(
        dataset=datasets["train"],
        key="maj",
        seed=hparams['seed']
    )

    if not HAS_brain.hparams.test_only:
        HAS_brain.fit(
            epoch_counter=HAS_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs={'sampler': train_sampler, 
                            **hparams["dataloader_options"]},
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    HAS_brain.hparams.num_samples = 100
    test_stats = HAS_brain.evaluate(
    test_set=datasets["test"],
    max_key="Acc",
    test_loader_kwargs={'batch_size':1},
    )

