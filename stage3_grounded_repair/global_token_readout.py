"""Token-local residual plus explicit latent-token transport, without teachers."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class GlobalTokenReadout(nn.Module):
    def __init__(self,input_rms,outputs=64,width=64,rank=32):
        super().__init__()
        if input_rms.ndim!=1 or not torch.isfinite(input_rms).all() or (input_rms<=0).any():
            raise ValueError('Positive finite TRAIN input RMS required')
        self.register_buffer('input_rms',input_rms.detach().clone().float())
        d=input_rms.numel();self.rank=rank
        self.local_input=nn.Linear(d,width);self.local_output=nn.Linear(width,outputs)
        self.query=nn.Linear(d,rank,bias=False);self.key=nn.Linear(d,rank,bias=False)
        self.transport_output=nn.Linear(outputs,outputs,bias=False)
        nn.init.zeros_(self.local_output.weight);nn.init.zeros_(self.local_output.bias)
        nn.init.zeros_(self.transport_output.weight)

    def forward(self,tokens,original_patches):
        if tokens.ndim!=3 or tokens.dtype!=torch.float32 or tokens.shape[-1]!=self.input_rms.numel():
            raise ValueError('Expected FP32 B,N,D visual features')
        if original_patches.ndim!=3 or original_patches.dtype!=torch.float32 or original_patches.shape[:2]!=tokens.shape[:2]:
            raise ValueError('Original physical latent patches must match each independent state')
        x=tokens/self.input_rms
        attention=torch.softmax((self.query(x)@self.key(x).transpose(-2,-1))/math.sqrt(self.rank),-1)
        transport=attention@original_patches-original_patches
        return self.local_output(F.silu(self.local_input(x)))+self.transport_output(transport)
