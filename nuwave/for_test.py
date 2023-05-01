from lightning_model import NuWave
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
from tqdm import tqdm
from scipy.io.wavfile import write as swrite
import json

def test(args):
    hparams = OC.load('hparameter.yaml')
    hparams.save = args.save or False
    model = NuWave(hparams, False).cuda()
    if args.ema:
        ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
                         
    else:
        ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}.ckpt'))[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
    print(ckpt_path)
    model.eval()
    model.freeze()
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)

    results=[]
    for i in range(5):
        snr=[]
        base_snr=[]
        lsd=[]
        base_lsd=[]
        t = model.test_dataloader()
        for j, batch in tqdm(enumerate(t), total=len(t)):
            wav, wav_l, name_wave = batch
            wav=wav.cuda()
            wav_l = wav_l.cuda()
            wav_up = model.sample(wav_l, model.hparams.ddpm.infer_step)
            snr.append(model.snr(wav_up,wav).detach().cpu())
            base_snr.append(model.snr(wav_l, wav).detach().cpu())
            lsd.append(model.lsd(wav_up,wav).detach().cpu())
            base_lsd.append(model.lsd(wav_l, wav).detach().cpu())
            if args.save and i==0:
                for l in range(len(batch[0])):
                    swrite(f'{hparams.log.test_result_dir}/test_{name_wave[l]}_up.wav',
                        hparams.audio.sr, wav_up[l].detach().cpu().numpy())
                    swrite(f'{hparams.log.test_result_dir}/test_{name_wave[l]}_orig.wav',
                        hparams.audio.sr, wav[l].detach().cpu().numpy())
                    swrite(f'{hparams.log.test_result_dir}/test_{name_wave[l]}_linear.wav',
                        hparams.audio.sr, wav_l[l].detach().cpu().numpy())
                    swrite(f'{hparams.log.test_result_dir}/test_{name_wave[l]}_down.wav',
                        hparams.audio.sr//hparams.audio.ratio, wav_l[l,::hparams.audio.ratio].detach().cpu().numpy())

        print(f'TEST loop {i} finished')
        snr = torch.stack(snr, dim =0).mean()
        base_snr = torch.stack(base_snr, dim =0).mean()
        lsd = torch.stack(lsd, dim =0).mean()
        base_lsd = torch.stack(base_lsd, dim =0).mean()
        d = {
            'snr': snr.item(),
            'base_snr': base_snr.item(),
            'lsd': lsd.item(),
            'base_lsd': base_lsd.item(),
        }
        print(d)
        with open(f'test_results_epochs_{args.resume_from}.json', mode='w') as outfile:
            json.dump({
                d
                }, outfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            default=356, help = "Resume Checkpoint epoch number")
    parser.add_argument('-e', '--ema',\
            required = False, default=True, help = "Start from ema checkpoint")
    parser.add_argument('--save', default = True,\
            help = "Save file")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    test(args)
