import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class EfficientSampler(nn.Module):

    def __init__(self, vocab_size: int):
        """ A more efficient sampler for large vocabularies.
         - uses two-stage hierarchical sampling to reduce computation

        Args:
            vocab_size (int): size of the vocabulary
        """
        super().__init__()

        self.vocab_size = vocab_size
        
        # break vocab into factors
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

    
    def forward(self, logits: torch.Tensor) -> torch.LongTensor:
        """ Sample from the logits using the efficient sampler.

        Args:
            logits (torch.Tensor): logits from the model [B, T, V]

        Returns:
            torch.Tensor: sampledd tokens [B, T]
        """
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


class LogMixture(nn.Module):

    def __init__(self, hidden_size, n, bias=0.0, mask=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.n = n
        self.bias = bias
        self.mask = mask

        self.mu = nn.Linear(hidden_size, n, bias=True)
        self.sigma = nn.Linear(hidden_size, n, bias=True)
        self.pi = nn.Linear(hidden_size, n, bias=True)
    

    def forward(self, hidden_states):
        mu = self.mu(hidden_states)
        sigma = self.sigma(hidden_states)
        pi = self.pi(hidden_states)
        
        sigma[..., 0] = sigma[..., 0] + self.bias
        pi[..., 0] = pi[..., 0] + self.mask

        sigma = sigma.exp()
        return LogMixtureDistribution(self.n, mu, sigma, pi)


class LogMixtureDistribution(nn.Module):
    def __init__(self, n, mu, sigma, pi):
        super().__init__()

        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

        print(self.log_mean())


    def log_mean(self):
        means = self.mu + self.sigma.pow(2)/2
        scales = torch.softmax(self.pi, -1)
        print(scales)

        return (means * scales).sum(-1)
    

    def log_prob(self, x, remove_last=False):
        mu = self.mu
        sigma = self.sigma
        pi = self.pi
        if remove_last:
            mu = mu[:, :-1]
            sigma = sigma[:, :-1]
            pi = pi[:, :-1]

        mu = mu.view(*x.shape, self.n)
        sigma = sigma.view(*x.shape, self.n)
        pi = pi.view(*x.shape, self.n)

        dist = torch.distributions.Normal(mu, sigma)
        logp = dist.log_prob(x.unsqueeze(-1))

        scales = F.log_softmax(pi, dim=-1)

        return torch.logsumexp(logp + scales, dim=-1)
