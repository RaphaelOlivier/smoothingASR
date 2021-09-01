from art.defences.preprocessor.mp3_compression import Mp3Compression

from torch.autograd import Function
import torch

def mp3_np(x_,fct):
    if x_.ndim==2 and x_.shape[0]==1: # batch size 1 and missing channel dim
        x_p = x_.reshape(1,-1,1)
        x_p,y_p = fct(x_p)
        x_p = x_p.reshape(1,-1)
    else:
        x_p,y_p = fct(x_)
    return x_p

class Mp3Pytorch(Function):
    @staticmethod
    def forward(
        ctx,
        x, fct):
        x_=x.clone().detach().cpu().numpy()

        x_p = mp3_np(x_,fct)
        x_p = torch.tensor(x_p, dtype=x.dtype,device=x.device)
        return x_p

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        return grad_input, None




class Mp3CompressionBaseline(Mp3Compression):
    
    def forward(self,x,y,*args,**kwargs):
        fct = super(Mp3CompressionBaseline,self).__call__
        if isinstance(x,torch.Tensor):
            x_p = x
            x_p += Mp3Pytorch.apply(x,fct)-x
        else:
            x_p = mp3_np(x,fct)
        return x_p,y

    def __call__(self,x,*args,**kwargs):
        return self.forward(x,*args,**kwargs)
