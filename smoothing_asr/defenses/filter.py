from art.defences.preprocessor import Preprocessor
from art.config import ART_NUMPY_DTYPE
import numpy as np
from audlib.enhance import wiener_iter, asnr, SSFEnhancer
from audlib.sig.window import hamming
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ASNRWiener(Preprocessor):
    def __init__(self,sr, hop, nfft,gaussian_sigma=None,high_freq=False, apply_fit: bool = True, apply_predict: bool = True):
        super(ASNRWiener,self).__init__()
        self.sr=sr
        self.window=hamming(nfft, hop=hop)
        self.hop=hop
        self.nfft=nfft
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.gaussian_sigma=gaussian_sigma
        self.lpc_order=12
        self.high_freq=high_freq

    @property
    def apply_fit(self) -> bool:
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        x_filtered=np.copy(x)
        for i in range(len(x)):
            if self.high_freq:
                noise = np.random.normal(0, scale=self.gaussian_sigma, size=(x[i].shape[0]+1,))
                noise = 0.5 * (noise[1:]-noise[:-1])
            else:
                noise = np.random.normal(0, scale=self.gaussian_sigma, size=x[i].shape).astype(ART_NUMPY_DTYPE) if self.gaussian_sigma else None
            filtered_output,_=asnr(x[i],self.sr, self.window, self.hop, self.nfft,noise=(noise if self.gaussian_sigma>0 else None),
            snrsmooth=0.98, noisesmooth=0.98, llkthres=.15, zphase=True, rule="wiener")
            if len(filtered_output)<len(x[i]):
                filtered_output = np.pad(filtered_output,mode="mean",pad_width=((0,len(x[i])-len(filtered_output))))
            elif len(filtered_output)>len(x[i]):
                filtered_output=filtered_output[:len(x[i])]
            x_filtered[i]=filtered_output
        return x_filtered, y

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass


class SSFPreprocessor(Preprocessor):
    def __init__(self,sr, hop, nfft, apply_fit: bool = True, apply_predict: bool = True):
        super(SSFPreprocessor,self).__init__()
        window=hamming(nfft, hop=hop)
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.enhancer=SSFEnhancer(sr, window, hop, nfft)
        self.lambda_lp=1e-3
    @property
    def apply_fit(self) -> bool:
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        x_filtered=np.copy(x)
        for i in range(len(x)):
            x_filtered[i]=self.enhancer(x[i],lambda_lp=self.lambda_lp)
        return x_filtered, y

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass


    