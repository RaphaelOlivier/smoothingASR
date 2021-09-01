from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocessor import Preprocessor
from art.config import ART_NUMPY_DTYPE
import numpy as np
from smoothing_asr.defenses.filter import ASNRWiener, SSFPreprocessor
import logging
from typing import Optional, Tuple, TYPE_CHECKING,Union
import torch
import torch.nn as nn
from torch.autograd import Function
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE
from torch.autograd import gradcheck
logger = logging.getLogger(__name__)

class SpeechNoiseAugmentation(GaussianAugmentation,nn.Module):
    def __init__(self,*args,high_freq=False, filter=None,filter_kwargs={},enhancer=None,enhancer_kwargs={},**kwargs):
        nn.Module.__init__(self)
        GaussianAugmentation.__init__(self,*args,**kwargs)
        self.filter=None
        self.enhancer=None
        self.high_freq=high_freq
        if filter is not None:
            if filter=="asnr_wiener":
                self.filter=ASNRWiener(gaussian_sigma=self.sigma,high_freq=high_freq, **filter_kwargs)
            elif filter=="ssf_enhancer":
                self.filter=SSFPreprocessor(**filter_kwargs)
            else:
                raise ValueError("Unrecognized filter %s"%filter)
        
        if enhancer is not None:
            from smoothing_asr.defenses.enhancer import SEGANEnhancer
            if enhancer=="segan":
                self.enhancer=SEGANEnhancer(**enhancer_kwargs)
            else:
                raise ValueError("Unrecognized enhancer %s"%enhancer)
    def forward(self, x: Union[torch.Tensor,np.ndarray], y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment the sample `(x, y)` with Gaussian noise. The result is either an extended dataset containing the
        original sample, as well as the newly created noisy samples (augmentation=True) or just the noisy counterparts
        to the original samples.
        """
        # Select indices to augment
        tensor_input = isinstance(x,torch.Tensor)
        if tensor_input:
            #x_enh = x
            x_enh = SmoothCh.apply(x,y,self.augmentation,self.ratio,self.sigma,self.high_freq,self.clip_values,self.filter,self.enhancer) #-x
            y_enh = None
        else:
            x_enh,y_enh = smooth_np(x,y,self.augmentation,self.ratio,self.sigma,self.high_freq,self.clip_values,self.filter,self.enhancer)
        return x_enh, y_enh

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)


def snr_db(x,x_aug):
    signal_power = (x ** 2).mean()
    noise_power = ((x - x_aug) ** 2).mean()
    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    return snr_db

def augment(x: np.ndarray,sigma,high_freq) -> np.ndarray:
    #x_aug = np.random.normal(x, scale=self.sigma, size=x.shape).astype(ART_NUMPY_DTYPE)
    x_aug = np.copy(x)
    for i in range(x.shape[0]):
        assert len(x[i].shape)==1
        if high_freq:
            noise = np.random.normal(0, scale=sigma, size=(x[i].shape[0]+1,))
            noise = 0.5 * (noise[1:]-noise[:-1])
        else:
            noise = np.random.normal(0, scale=sigma, size=x[i].shape)
        x_aug[i] = (x[i]+noise).astype(ART_NUMPY_DTYPE)
    return x_aug


def smooth_np(
    x: np.ndarray, 
    y: Optional[np.ndarray],
    augmentation:bool,
    ratio:float,
    sigma:float,
    high_freq:bool,
    clip_values:tuple,
    filter,
    enhancer
    ):
    if augmentation:
        logger.info("Original dataset size: %d", x.shape[0])
        size = int(x.shape[0] * ratio)
        indices = np.random.randint(0, x.shape[0], size=size)

        # Generate noisy samples
        x_aug = augment(x[indices],sigma,high_freq)
        x_aug = np.vstack((x, x_aug))
        if y is not None:
            y_aug = np.concatenate((y, y[indices]))
        else:
            y_aug = y
        logger.info("Augmented dataset size: %d", x_aug.shape[0])
    else:
        x_aug = augment(x,sigma,high_freq)
        y_aug = y

    if clip_values is not None:
        x_aug = np.clip(x_aug, clip_values[0], clip_values[1])
    #print("SNR :",[snr_db(x[i],x_aug[i]) for i in range(len(x))])
    if filter:
        x_filt, y_filt=filter(x_aug, y_aug)
    else:
        x_filt, y_filt=x_aug, y_aug
    if enhancer:
        x_enh, y_enh=enhancer(x_filt, y_filt)
    else:
        x_enh, y_enh=x_filt, y_filt

    return x_enh,y_enh


class SmoothCh(Function):
    
    @staticmethod
    def forward(
        ctx,
        x,
        y,
        augmentation,
        ratio,
        sigma,
        high_freq,
        clip_values,
        filter,
        enhancer):
        x_=x.clone()
        x_np=x_.detach().cpu().numpy()
        x_enh, y_enh = smooth_np(x_np,y,augmentation,ratio,sigma,high_freq,clip_values,filter,enhancer)
        x_enh = torch.tensor(x_enh).to(x_.device)
        return x_enh

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        return grad_input, None, None, None, None, None, None, None, None


class Identity(Function):
    @staticmethod
    def forward(
        ctx,
        x):
        x_=x.clone()
        return x_

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        return grad_input

class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None