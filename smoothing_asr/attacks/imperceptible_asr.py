from __future__ import absolute_import, division, print_function, unicode_literals

# version of the attack that runs with smoothed model (and its preprocessing defenses)

import logging
from typing import Tuple, Optional, Union, TYPE_CHECKING

import numpy as np
import scipy
from art.config import ART_NUMPY_DTYPE

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
logger = logging.getLogger(__name__)


class ImperceptibleASRSmoothedGradients(ImperceptibleASRPyTorch):
    def __init__(
        self,
        *args,
        niters_gradients:int = 1,
        second_stage=False,
        **kwargs
    ):
        super(ImperceptibleASRSmoothedGradients,self).__init__(*args,**kwargs)
        self.niters_gradients=niters_gradients
        self.second_stage=second_stage

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        original_output: np.ndarray,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray, "torch.Tensor", "torch.Tensor"]:
        """
        The forward pass of the first stage of the attack.

        :param original_input: Samples of shape (nb_samples, seq_length). Note that, sequences in the batch must have
                               equal lengths. A possible example of `original_input` could be:
                               `original_input = np.array([np.array([0.1, 0.2, 0.1]), np.array([0.3, 0.1, 0.0])])`.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :param local_batch_size: Current batch size.
        :param local_max_length: Max length of the current batch.
        :param rescale: Current rescale coefficients.
        :param input_mask: Masks of true inputs.
        :param real_lengths: Real lengths of original sequences.
        :return: A tuple of (loss, local_delta, decoded_output, masked_adv_input)
                    - loss: The loss tensor of the first stage of the attack.
                    - local_delta: The delta of the current batch.
                    - decoded_output: Transcription output.
                    - masked_adv_input: Perturbed inputs.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.nn import CTCLoss

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(self.estimator.device)
        local_delta_rescale *= torch.tensor(rescale).to(self.estimator.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.estimator.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.estimator.device)

        # batch data for EoT gradient
        masked_adv_input_batch = masked_adv_input.expand(self.niters_gradients,masked_adv_input.size(1)).clone()
        input_mask=np.repeat(input_mask,self.niters_gradients,0)
        real_lengths=np.repeat(real_lengths,self.niters_gradients,0)
        original_output = np.repeat(original_output,self.niters_gradients,0)
        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, batch_idx = self.estimator.preprocess_transform_model_input(
            x=masked_adv_input_batch.to(self.estimator.device),
            y=original_output,
            real_lengths=real_lengths,
        )


        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.estimator.model(
            inputs.to(self.estimator.device), input_sizes.to(self.estimator.device)
        )
        outputs_ = outputs.transpose(0, 1)
        float_outputs = outputs_.float()

        # Loss function
        criterion = CTCLoss()
        loss = criterion(float_outputs.log_softmax(-1), targets, output_sizes, target_sizes).to(self.estimator.device)
        loss = loss / inputs.size(0)
        # Compute transcription
        decoded_output, _ = self.estimator.decoder.decode(F.softmax(outputs,-1), output_sizes)
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)
        # Rearrange to the original order
        decoded_output_ = decoded_output.copy()
        decoded_output[batch_idx] = decoded_output_

        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale

    def _generate_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: A batch of adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # First stage of attack
        successful_adv_input_1st_stage, original_input = self._attack_1st_stage(x=x, y=y)

        results = successful_adv_input_1st_stage.detach().cpu().numpy()


        if self.second_stage:
            successful_perturbation_1st_stage = successful_adv_input_1st_stage - torch.tensor(original_input).to(
                self.estimator.device
            )

            # Compute original masking threshold and maximum psd
            theta_batch = []
            original_max_psd_batch = []

            for i in range(len(x)):
                theta, original_max_psd = self._compute_masking_threshold(original_input[i])
                theta = theta.transpose(1, 0)
                theta_batch.append(theta)
                original_max_psd_batch.append(original_max_psd)

            theta_batch = np.array(theta_batch)
            original_max_psd_batch = np.array(original_max_psd_batch)

            # Reset delta with new result
            local_batch_shape = successful_adv_input_1st_stage.shape
            self.global_optimal_delta.data = torch.zeros(self.batch_size, self.global_max_length).type(torch.float64)
            self.global_optimal_delta.data[
                : local_batch_shape[0], : local_batch_shape[1]
            ] = successful_perturbation_1st_stage

            # Second stage of attack
            successful_adv_input_2nd_stage = self._attack_2nd_stage(
                x=x, y=y, theta_batch=theta_batch, original_max_psd_batch=original_max_psd_batch
            )

            results = successful_adv_input_2nd_stage.detach().cpu().numpy()

        return results
