import os
import torch
import torch.nn as nn
import numpy as np
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import Preprocessor
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from armory import paths
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)

class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

class SEGANEnhancer(Preprocessor):
    def __init__(self,load_weights_file,cfg_file,  apply_fit: bool = True, apply_predict: bool = True):
        from segan.models import *
        saved_model_dir = paths.runtime_paths().saved_model_dir
        model_path = os.path.join(saved_model_dir, load_weights_file)
        opts_path = os.path.join(saved_model_dir, cfg_file)
        with open(opts_path, 'r') as cfg_f:
            args = ArgParser(json.load(cfg_f))
        args.cuda = torch.cuda.is_available()
        self.device="cuda" if args.cuda else "cpu"
        if hasattr(args, 'wsegan') and args.wsegan:
            self.model = WSEGAN(args)     
        else:
            self.model = SEGAN(args)     

        self.model.G.load_pretrained(model_path, True)
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict


    def __call__(self,x: np.ndarray, y: Optional[np.ndarray] = None):
        x_enhanced = np.copy(x)
        for i in range(len(x)):
            segan_input = torch.tensor(x[i].reshape(1,1,-1))
            segan_output, _ = self.model.generate(segan_input)
            x_enhanced[i]=segan_output
        return x_enhanced, y

    
    @property
    def apply_fit(self) -> bool:
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass