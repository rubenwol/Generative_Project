from metrics import OurMetrics
import os
import torchaudio
import torch
import json
from tqdm import tqdm

def evaluation(output_path,
               clean_path='/home/nlp/wolhanr/NuWave/VCTK-DEMAND/test/clean',
               noisy_path='/home/nlp/wolhanr/NuWave/VCTK-DEMAND/test/noisy',
               dataset='cmgan'):

    cut_len=16000*2
    list_files = os.listdir(clean_path)
    metrics = OurMetrics()
    snr = []
    base_snr = []
    lsd = []
    base_lsd = []
    error = 0
    file_not_found = []
    for file in tqdm(list_files):
        output_file = os.path.join(output_path, file)
        clean_file = os.path.join(clean_path, file)
        noisy_file = os.path.join(noisy_path, file)
        if dataset == 'nuwave':
            output_file = os.path.join(output_path, f'test_{file}_up.wav')

        if dataset == 'wavnet':
            file_name = file.split('.')[0]
            output_file = os.path.join(output_path, f'{file_name}_denoised.wav')
            clean_file = os.path.join(output_path, f'{file_name}_clean.wav')
            noisy_file = os.path.join(output_path, f'{file_name}_noisy.wav')
        try:
            clean_ds, _ = torchaudio.load(clean_file)
            noisy_ds, _ = torchaudio.load(noisy_file)
            output_ds, _ = torchaudio.load(output_file)
        except:
            file_not_found.append(file)
            continue

        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        output_ds = output_ds.squeeze()

        try:
            snr.append(metrics.snr(output_ds, clean_ds))
            lsd.append(metrics.lsd(output_ds, clean_ds))
            base_snr.append(metrics.snr(noisy_ds, clean_ds))
            base_lsd.append(metrics.lsd(noisy_ds, clean_ds))
        except:
            error += 1

    snr = torch.stack(snr, dim =0).mean()
    base_snr = torch.stack(base_snr, dim =0).mean()
    lsd = torch.stack(lsd, dim =0).mean()
    base_lsd = torch.stack(base_lsd, dim =0).mean()
    d = {
        'snr': snr.item(),
        'base_snr': base_snr.item(),
        'lsd': lsd.item(),
        'base_lsd': base_lsd.item(),
        'number_error': error,
        'number_file_not_found': len(file_not_found)
          }


    with open(f'results_{dataset}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(d))
    print(d)
    print(file_not_found)

if __name__ == '__main__':
    #evaluation('/home/nlp/wolhanr/CMGAN/src/saved_tracks_best')
    #evaluation('/home/nlp/wolhanr/NuWave/nuwave/test_sample/result', dataset = 'nuwave')
    evaluation('/home/nlp/wolhanr/CMGAN/wavenet_output', dataset = 'wavnet')



