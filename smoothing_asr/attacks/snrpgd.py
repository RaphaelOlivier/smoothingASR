import logging
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from art.attacks.attack import EvasionAttack

import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
logger = logging.getLogger(__name__)

class PGDSNR(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "snrdb_radius",
        "step_factor",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "max_iter",
        "random_eps",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)


    def __init__(self,estimator,*args,snrdb_radius,step_factor,**kwargs):
        super().__init__(estimator=estimator)
        self.args=args 
        self.kwargs=kwargs 
        self.snrdb_radius=snrdb_radius
        self.radius_coeff = np.power(10,-snrdb_radius/20)
        logger.info("Radius coefficient with provided SNR : %f"%self.radius_coeff)
        self.step_factor=step_factor
    
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        assert x.shape[0]==1
        logger.info("Computing noise radius")
        normx = np.linalg.norm(x)
        eps = self.radius_coeff*normx
        logger.info("Radius %f"%eps)
        eps_step = eps*self.step_factor
        self._attack = ProjectedGradientDescent(self.estimator,*self.args,eps=eps,eps_step=eps_step,**self.kwargs)
        logger.info("Creating adversarial samples.")
        y = y.astype(str)
        x_adv = self._attack.generate(x=x, y=y, **kwargs)
        return x_adv

