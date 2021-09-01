from deepspeech_pytorch.decoder import Decoder, GreedyDecoder, BeamCTCDecoder

import Levenshtein as Lev
import torch
from six.moves import xrange

def load_decoder_with_scores(decoder):
    # make BeamDecoder also return scores
    if isinstance(decoder,BeamCTCDecoder):
        return BeamCTCDecoderWithScores(decoder)
    else:
        assert isinstance(decoder,GreedyDecoder)
        return GreedyDecoderWithScores(decoder)

class BeamCTCDecoderWithScores(BeamCTCDecoder):

    def __init__(self,decoder:BeamCTCDecoder):
        for attr,value in decoder.__dict__.items():
            setattr(self,attr,value)
    def decode(self, probs, sizes=None,return_scores=False):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)
        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        if return_scores:
            return strings, offsets, scores
        else:
            return strings, offsets


class GreedyDecoderWithScores(GreedyDecoder):

    def __init__(self,decoder:GreedyDecoder):
        for attr,value in decoder.__dict__.items():
            print(attr)
            setattr(self,attr,value)
    
    def decode(self, probs, sizes=None,return_scores=False):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        max_probs, max_probs_idx = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs_idx.view(max_probs_idx.size(0), max_probs_idx.size(1)),
                                                    sizes,
                                                    remove_repetitions=True,
                                                    return_offsets=True)
        max_logits=torch.log(max_probs)
        batch_size=len(sizes)
        scores = [max_logits[k,:sizes[k]] for k in range(batch_size)]
        if return_scores:
            return strings, offsets, scores
        else:
            return strings, offsets