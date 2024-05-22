import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class EfficientSampler(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size
        
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

    
    def forward(self, logits):
        assert logits.dim() == 3
        assert logits.size()[-1] == self.vocab_size
        batch_size, seq_length = logits.shape[:2]

        # get prob rectangle
        factored_probs = torch.softmax(
            logits.detach().float(), dim=-1
        ).view(-1, self.vocab_factor, self.vocab_chunk)

        # sample row
        outer_probs = factored_probs.sum(dim=-1)
        outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

        # sample column
        ar = torch.arange(batch_size*seq_length, device=logits.device, dtype=torch.long)
        inner_probs = factored_probs[ar, outer_sample]
        inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

        # combine and reshape
        sample = (self.vocab_chunk*outer_sample + inner_sample)
        return sample.view(batch_size, seq_length)
    