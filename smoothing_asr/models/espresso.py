"""
Automatic speech recognition model
"""
import os
import numpy as np
import torch
from armory import paths
from art.estimators.speech_recognition import PyTorchEspresso
from art.utils import get_file
from deepspeech_pytorch.utils import load_model
from art.config import ART_DATA_PATH
from smoothing_asr.models.vote import MajorityVote, Rover, MultipleProbsVote, VoteEnsemble, ROVER_MAX_HYPS
from smoothing_asr.models.decoder import load_decoder_with_scores
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import torch.nn.functional as F
from art import config

if TYPE_CHECKING:
    import torch

import art 
logger = logging.getLogger(__name__)

import torch  # lgtm [py/repeated-import]
import yaml
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import encoders
import sentencepiece as spm
import ast
from argparse import Namespace
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
#from espresso.tasks.speech_recognition import SpeechRecognitionEspressoTask, SpeechRecognitionEspressoConfig
class SmoothedEspresso(PyTorchEspresso):
    def __init__(
        self,
        model,
        *args,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
        verbose: bool = True,
        voting_kwargs=None, 
        niters_forward=1, 
        niters_backward=1,
        batch_backward=0,
        batch_forward=0,
        **kwargs):
        #optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4, weight_decay=1e-5, amsgrad=False)
        model_dir = paths.runtime_paths().saved_model_dir
        data_dir = paths.runtime_paths().dataset_dir
        config_path = os.path.join(model_dir,"libri960_transformer.yaml")
        model_path = os.path.join(model_dir,"transformer.checkpoint_best.pt")
        sp_path = os.path.join(model_dir,"train_960_unigram5000.model")
        dict_path = os.path.join(model_dir,"train_960_unigram5000_units.txt")
        data_path = os.path.join(data_dir,"librispeech","plain_text","1.1.0","dataset_info")
        state_dict = torch.load(model_path)
        super(PyTorchEspresso,self).__init__(
            model=None,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.espresso_config_filepath = config_path
        self.verbose = verbose
        cuda_idx = torch.cuda.current_device()
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        with open(config_path) as file:
            esp_args_dict = yaml.load(file, Loader=yaml.FullLoader)
            esp_args = Namespace(**esp_args_dict)
            esp_args.path = model_path
            esp_args.sentencepiece_model = sp_path
            esp_args.dict = dict_path
        self.esp_args = esp_args

        # setup espresso/fairseq task
        #self.task = setup_task_without_dataset(SpeechRecognitionEspressoTask,self.esp_args)
        self.task = tasks.setup_task(self.esp_args)
        self.task.feat_dim = self.esp_args.feat_dim

        # load_model_ensemble
        self._models, self._model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.esp_args.path),
            arg_overrides=ast.literal_eval(self.esp_args.model_overrides),
            task=self.task,
            suffix=getattr(self.esp_args, "checkpoint_suffix", ""),
        )
        for m in self._models:
            m.to(self._device)

        self._model = self._models[0]

        self.dictionary = self.task.target_dictionary
        self.generator = self.task.build_generator(self._models, self.esp_args)
        self.tokenizer = encoders.build_tokenizer(self.esp_args)
        self.bpe = encoders.build_bpe(self.esp_args)  # bpe encoder
        self.spp = spm.SentencePieceProcessor()  # sentence piece model
        self.spp.Load(self.esp_args.sentencepiece_model)

        self.criterion = self.task.build_criterion(self.esp_args)
        self._sampling_rate = self.esp_args.sampling_rate

        self.niters_forward=niters_forward
        self.niters_backward= niters_backward
        #self.decoder = load_decoder_with_scores(self.decoder)
        self.set_voting_module(**voting_kwargs, **kwargs)
        self.batch_backward=batch_backward
        self.batch_forward=batch_forward
        self.criterion.print_interval= 1e6
        delattr(self._model,"num_updates")

    def set_voting_module(self,voting="majority",rover_bin_path=None, vote_on_nbest=False, decoder_type="greedy", use_alignments=False,use_confidence=False, **kwargs):
        assert (not vote_on_nbest) or (decoder_type=="beam"), "option vote_on_nbest is incompatible with greedy decoding"
        assert (not use_confidence) or decoder_type=="beam" or voting=="majority", "use_confidence is currently not compatible with greedy search. You can use beam search with width 1"
        if voting in ["rover","rover_freq"]: # Rover by Word frequency
            if self.niters_forward>ROVER_MAX_HYPS:
                self.voting_module = VoteEnsemble(Rover(scheme='freq', exec_path=rover_bin_path, return_all=True),Rover(scheme='freq', exec_path=rover_bin_path))
            else:
                self.voting_module=Rover(scheme='freq', exec_path=rover_bin_path)
        elif voting=="rover_conf": # Rover by Average Confidence Scores 
            self.voting_module=Rover(scheme='conf', exec_path=rover_bin_path)
        elif voting=="rover_max": # Rover by Word Maximum Confidence Scores 
            self.voting_module=Rover(scheme='max', exec_path=rover_bin_path)
        elif voting=="probs_sum":
            raise ValueError("probs voting mode not available with transformer model")
            #self.voting_module=MultipleProbsVote(scheme="sum", decoder=self.decoder, device=self._device)
        elif voting=="probs_max":
            raise ValueError("probs voting mode not available with transformer model")
            #self.voting_module=MultipleProbsVote(scheme="max", decoder=self.decoder, device=self._device)
        elif voting=="probs_rover":
            raise ValueError("probs voting mode not available with transformer model")
            #self.voting_module = VoteEnsemble(MultipleProbsVote(scheme="sum", decoder=self.decoder, device=self._device, return_all = True), 
            #Rover(scheme='freq', exec_path=rover_bin_path), agg_by=12)
        else:
            assert voting=="majority"
            self.voting_module=MajorityVote()

        self.transcription_output = not voting.startswith("probs")
        self.use_alignments=use_alignments
        self.use_confidence=use_confidence
    
    def predict(self,x: np.ndarray,reload_model=False,**kwargs):
        assert not reload_model
        if self.niters_forward<=1:
            decoded_output = self.predict_once(x, **kwargs)
        else:
            random_outputs=[]
            alignments=[]
            confidence=[]
            kwargs["transcription_output"]=self.transcription_output
            kwargs["return_alignments"]=True
            kwargs["return_confidence"]=True
            if self.batch_forward>0 and x.shape[0]==1:
                for i in range(0,self.niters_forward,self.batch_forward):
                    batch_size = min(self.batch_forward,self.niters_forward-i)
                    x_batch = np.repeat(x,batch_size,axis=0)
                    out=self.predict_once(x_batch, **kwargs)
                    out, algns ,scores = out
                    if self.transcription_output:
                        for j in range(batch_size):
                            alignments.append(algns[j].reshape(1,-1))
                            confidence.append(scores[j].reshape(1,-1))
                            random_outputs.append(out[j].reshape(1))
                    else:
                        alignments.append(None)
                        confidence.append(None)
                        out = list(zip(*out))
                        for j in range(batch_size):
                            o,s = out[j]
                            o=o.reshape(1,*o.shape)
                            s=s.reshape(1,*s.shape)
                            random_outputs.append((o,s))

            else:
                for i in range(self.niters_forward):
                    out=self.predict_once(x, **kwargs)
                    out, algns ,scores = out
                    alignments.append(algns)
                    confidence.append(scores)
                    random_outputs.append(out)
            decoded_output=self.voting_module.run(
                asr_outputs=random_outputs, 
                alignments=alignments if self.use_alignments else None, 
                confidence=confidence if self.use_confidence else None,
                char_alignment=False
            )

        decoded_output = np.array([self.bpe.decode(hypo_str) for hypo_str in decoded_output])
        return decoded_output


    def predict_once(
        self, x: np.ndarray, batch_size: int = 128, return_nbest=False,return_alignments=False, return_confidence=False, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :return: Transcription as a numpy array of characters. A possible example of a transcription return
                 is `np.array(['SIXTY ONE', 'HELLO'])`.
        """

        def get_symbols_to_strip_from_output(generator):
            if hasattr(generator, "symbols_to_strip_from_output"):
                return generator.symbols_to_strip_from_output

            return {generator.eos, generator.pad}

        x_in = np.empty(len(x), dtype=object)
        x_in[:] = list(x)

        # Put the model in the eval mode
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_in, y=None, fit=False)

        # Run prediction with batch processing
        decoded_output = []
        # result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=np.int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Transform x into the model input space
            # Note that batch is re-ordered during transformation
            batch, batch_idx = self._transform_model_input(x=x_preprocessed[begin:end])

            hypos = self.task.inference_step(self.generator, self._models, batch)
            alignments = []
            scores = []
            for _, hypos_i in enumerate(hypos):
                # Process top predictions
                for _, hypo in enumerate(hypos_i[: self.esp_args.nbest]):
                    hypo_str = self.dictionary.string(
                        hypo["tokens"].int().cpu(),
                        bpe_symbol=None,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                    )  # not removing bpe at this point
                    detok_hypo_str = self.bpe.decode(hypo_str)
                    decoded_output.append(hypo_str)
                    algn_feats = torch.max(hypo["attention"],dim=0)[1]
                    frames_feats_ratio = len(x_preprocessed[0])/len(hypo["attention"])
                    algn_frames  =(frames_feats_ratio*algn_feats).cpu().numpy()
                    alignments.append(algn_frames)
                    scores.append(algn_frames) # not used

        decoded_output_array = np.array(decoded_output)
        decoded_output_copy = decoded_output_array.copy()
        decoded_output_array[batch_idx] = decoded_output_copy  # revert decoded output to its original order
        
        if return_alignments or return_confidence:
            results=[decoded_output_array] + ([alignments] if return_alignments else []) + ([scores] if return_confidence else [])
            return tuple(results)
        return decoded_output_array
    
    
    def loss_gradient(self,x: np.ndarray, y: np.ndarray,**kwargs):
        if self.niters_backward<=1:
            return self.loss_gradient_no_batch(x,y,**kwargs)
        else:
            random_gradients=[]
            if self.batch_backward>0 and x.shape[0]==1:
                for i in range(0,self.niters_backward,self.batch_backward):
                    batch_size = min(self.batch_forward,self.niters_backward-i)
                    x_batch = np.repeat(x,batch_size,axis=0)
                    y_batch = np.repeat(y,batch_size,axis=0)
                    grad=self.loss_gradient_no_batch(x_batch,y_batch,**kwargs)
                    random_gradients=random_gradients + [np.expand_dims(g,0) for g in grad]
            else:
                for i in range(self.niters_backward):
                    grad=self.loss_gradient_no_batch(x,y,**kwargs)
                    random_gradients.append(grad)
            result_gradient=self.estimate_input_gradient(random_gradients)
            return result_gradient


    def loss_gradient_no_batch(self,x: np.ndarray, y: np.ndarray,**kwargs):
        x_in = np.empty(len(x), dtype=object)
        x_in[:] = list(x)

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self._model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_in, y, fit=False)

        # Transform data into the model input space
        batch_dict, batch_idx = self._transform_model_input(x=x_preprocessed, y=y_preprocessed, compute_gradient=True)

        loss, _, _ = self.criterion(self._model, batch_dict)
        loss.backward()

        # Get results
        results_list = list()
        src_frames = batch_dict["net_input"]["src_tokens"].grad.cpu().numpy().copy()
        src_lengths = batch_dict["net_input"]["src_lengths"].cpu().numpy().copy()
        for i, _ in enumerate(x_preprocessed):
            results_list.append(src_frames[i, : src_lengths[i], :])

        results = np.array(results_list)

        if results.shape[0] == 1:
            results_ = np.empty(len(results), dtype=object)
            results_[:] = list(results)
            results = results_

        # Rearrange to the original order
        results_ = results.copy()
        results[batch_idx] = results_

        results = self._apply_preprocessing_gradient(x_in, results)

        if x.dtype != np.object:
            results = np.array([i for i in results], dtype=x.dtype)  # pylint: disable=R1721
            results = results.reshape(x.shape)
            #assert results.shape == x.shape and results.dtype == x.dtype
        else:
            results = np.array([np.squeeze(res) for res in results], dtype=np.object)

        # Unfreeze batch norm layers again
        self.set_batchnorm(train=True)

        return results

    def estimate_input_gradient(self,gradients_list):
        nsamples=len(gradients_list)
        gradient=1./nsamples*sum(gradients_list)
        return gradient
    
    def compute_loss(self):
        raise NotImplementedError

    def _preprocess_transform_model_input(
        self,
        x: "torch.Tensor",
        y: Optional[np.ndarray]=None,
        tensor_input=True,
        real_lengths: Optional[np.ndarray]=None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", List]:
        import torch  # lgtm [py/repeated-import]
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        # Transform the input space
        inputs, targets, input_rates, target_sizes, batch_idx = self._transform_model_input(
            x=x_preprocessed,
            y=y,
            compute_gradient=False,
            tensor_input=tensor_input,
            real_lengths=real_lengths,
        )

        return inputs, targets, input_rates, target_sizes, batch_idx

    def _transform_model_input(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Optional[np.ndarray] = None,
        compute_gradient: bool = False,
    ) -> Tuple[Dict, List]:
        """
        Transform the user input space into the model input space.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :return: A tuple of a dictionary of batch and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        from fairseq.data import data_utils

        def _collate_fn(batch: List) -> dict:
            """
            Collate function that transforms a list of numpy array or torch tensor representing a batch into a
            dictionary that Espresso takes as input.
            """
            # sort by seq length in descending order
            batch = sorted(batch, key=lambda t: t[0].size(0), reverse=True)
            batch_size = len(batch)
            max_seqlength = batch[0][0].size(0)
            src_frames = torch.zeros(batch_size, max_seqlength, 1)
            src_lengths = torch.zeros(batch_size, dtype=torch.long)

            for i, (sample, _) in enumerate(batch):
                seq_length = sample.size(0)
                src_frames[i, :seq_length, :] = sample.unsqueeze(1)
                src_lengths[i] = seq_length

            if compute_gradient:
                src_frames = torch.tensor(src_frames,device=self._device, requires_grad=True)
                src_frames.requires_grad = True

            # for input feeding in training
            if batch[0][1] is not None:
                pad_idx = self.dictionary.pad()
                eos_idx = self.dictionary.eos()
                target = data_utils.collate_tokens(
                    [s[1] for s in batch],
                    pad_idx,
                    eos_idx,
                    False,
                    False,
                    pad_to_length=None,
                    pad_to_multiple=1,
                )
                prev_output_tokens = data_utils.collate_tokens(
                    [s[1] for s in batch],
                    pad_idx,
                    eos_idx,
                    False,
                    True,
                    pad_to_length=None,
                    pad_to_multiple=1,
                )
                target = target.long().to(self._device)
                prev_output_tokens = prev_output_tokens.long().to(self._device)
                ntokens = sum(s[1].ne(pad_idx).int().sum().item() for s in batch)

            else:
                target = None
                prev_output_tokens = None
                ntokens = None

            batch_dict = {
                "ntokens": ntokens,
                "net_input": {
                    "src_tokens": src_frames.to(self._device),
                    "src_lengths": src_lengths.to(self._device),
                    "prev_output_tokens": prev_output_tokens,
                },
                "target": target,
            }

            return batch_dict

        # We must process each sequence separately due to the diversity of their length
        batch = []
        for i, _ in enumerate(x):
            # First process the target
            if y is None:
                target = None
            else:
                eap = self.spp.EncodeAsPieces(y[i])
                sp_string = " ".join(eap)
                target = self.dictionary.encode_line(sp_string, add_if_not_exist=False)  # target is a long tensor

            # Push the sequence to device
            if isinstance(x, np.ndarray):
                x[i] = x[i].astype(config.ART_NUMPY_DTYPE)
                x[i] = torch.tensor(x[i]).to(self._device)

            # Set gradient computation permission
            if compute_gradient:
                x[i].requires_grad = True

            # Re-scale the input audio to the magnitude used to train Espresso model
            x[i] = x[i] * 32767

            # Form the batch
            batch.append((x[i], target))

        # We must keep the order of the batch for later use as the following function will change its order
        batch_idx = sorted(range(len(batch)), key=lambda i: batch[i][0].size(0), reverse=True)

        # The collate function is important to convert input into model space
        batch_dict = _collate_fn(batch)

        # return inputs, targets, input_percentages, target_sizes, batch_idx
        return batch_dict, batch_idx

def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    return SmoothedEspresso(**wrapper_kwargs)

