#!/usr/bin/env python3
"""
Code for training I-CNF

> python3 Train_I-CNF.py Train_I-CNF.yaml
"""
import os
import sys
import time
import torch
import random
import ruamel.yaml
import numpy as np
import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask
from hyperpyyaml import load_hyperpyyaml
import normflows as nf
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
    label_dic = np.load(hparams['label_file'],allow_pickle=True).item()
    fea_dic = np.load(hparams['fea_path'],allow_pickle=True).item()
    
    # Define data pipeline
    @sb.utils.data_pipeline.takes("utteranceId")
    @sb.utils.data_pipeline.provides("sig","label")
    def audio_pipeline(utteranceId):

        utt_name=utteranceId.split('.')[0]
        sig = fea_dic[utt_name]
        yield sig
        lab = label_dic[utt_name]
        yield torch.from_numpy(lab).float()

    # Define datasets:
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig","duration","label"],
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


# Brain class for conditional integer flows
class ICNF_brain(sb.Brain):
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
        if isinstance(label,sb.dataio.batch.PaddedData):
            label=label.data

        label_ref = torch.stack([torch.mean(x,dim=0) for x in label])
        num_rater = torch.tensor([len(x) for x in label]).to(self.device)
        label_mask = get_mask_from_lengths(num_rater)
        label_padded = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0.0)
        assert(label_padded.shape[0] == label_mask.shape[0] and label_padded.shape[1] == label_mask.shape[1]),(label_padded.shape,label_mask.shape)

        return label_ref,label_padded,label_mask,num_rater.unsqueeze(-1)
    

    def compute_objectives(self, predictions, batch, stage):
        label_ref,label,label_mask, rater_lens=self.process_label(batch.to(self.device)) 
        # label.shape: [B, num_rater, output_dim]
        # label_ref.shape: [B, output_dim]
        # label_mask.shape: [B, num_rater]

        loc, log_scale = predictions

        if self.hparams.output_dim ==1:
            label=label.unsqueeze(-1)
            label_ref=label_ref.unsqueeze(-1)
        
        loss = self.modules.flow.forward_kld(label, loc, log_scale) 
        loss = loss.masked_fill_(label_mask, 0.0)
        loss_NLL_all = torch.sum(loss,dim=-1)/rater_lens.squeeze(-1)
        loss_NLL_avg = self.modules.flow.forward_kld(label_ref, loc, log_scale)
        loss=loss_NLL_all.mean()

        z, _, z_latent = self.modules.flow.sample(loc,log_scale, num_samples=self.hparams.num_samples,return_latent=True)
        z_mean = torch.mean(z,dim=1)
        z_std = torch.std(z,dim=1)
        self.mse_metric.append(batch.id, targets=label_ref, predictions=z_mean,reduction='batch')
        
        if stage == sb.Stage.TEST:
            for i in range(len(batch.id)):
                self.test_outcome[batch.id[i]]=z[i].detach().cpu().numpy()
                self.test_NLL_all[batch.id[i]]=loss_NLL_all[i].detach().cpu().numpy()
                self.test_NLL_avg[batch.id[i]]=loss_NLL_avg[i].detach().cpu().numpy()
        return loss


    def on_stage_start(self, stage, epoch=None):
        self.start_time = time.time()
        self.mse_metric =self.hparams.error_stats_mse()
        
        if stage == sb.Stage.TEST:
            self.test_outcome = {}
            self.test_NLL_all={}
            self.test_NLL_avg={}
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        self.elapse_time = time.time() - self.start_time

        rmse=np.sqrt(self.mse_metric.summarize('average'))
        stats = {"loss": stage_loss,
                 'RMSE': rmse,
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
                                                 min_keys=["loss"], 
                                                 num_to_keep=3, 
                                                 name=f"E{self.hparams.epoch_counter.current}")

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            np.save(os.path.join(self.hparams.output_folder,'test_outcome-E{}.npy'.format(self.hparams.epoch_counter.current)),self.test_outcome)
            np.save(os.path.join(self.hparams.output_folder,'test_NLL_all-E{}.npy'.format(self.hparams.epoch_counter.current)),self.test_NLL_all)    
            np.save(os.path.join(self.hparams.output_folder,'test_NLL_avg-E{}.npy'.format(self.hparams.epoch_counter.current)),self.test_NLL_avg)
    
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

    HAS_brain = ICNF_brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    HAS_brain.checkpointer.add_recoverable("nfm", nfm)

    
    if not HAS_brain.hparams.test_only:
        HAS_brain.fit(
            epoch_counter=HAS_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    
    HAS_brain.hparams.num_samples = 100
    test_stats = HAS_brain.evaluate(
    test_set=datasets["test"],
    min_key="RMSE",
    test_loader_kwargs={'batch_size':1},
    )
