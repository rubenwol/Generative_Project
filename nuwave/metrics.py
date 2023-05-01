import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    #x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        mag = torch.norm(stft, p=2, dim =-1) #[B, F, TT]
        return mag

class OurMetrics:
    def __init__(self):
        self.stft = STFTMag(2048, 512)


        def snr(pred, target):
            return (20 *torch.log10(torch.norm(target, dim=-1) \
                    /torch.norm(pred -target, dim =-1).clamp(min =1e-8))).mean()

        def lsd(pred, target):
            stft = STFTMag(2048, 512)
            sp = torch.log10(stft(pred).square().clamp(1e-8))
            st = torch.log10(stft(target).square().clamp(1e-8))
            return (sp - st).square().mean(dim=1).sqrt().mean()

        self.snr = snr
        self.lsd = lsd


    