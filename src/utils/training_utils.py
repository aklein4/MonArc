import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(logits, x, tokenizer):
    x, logits = x[:, 1:], logits[:, :-1]

    return F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        ignore_index=tokenizer.pad_token_id
    )


@torch.no_grad()
def ppl(logits, x, tokenizer):
    x = x[:, 1:]
    logits = logits[:, :-1]
    mask = x != tokenizer.pad_token_id

    logp = -F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        reduction='none'
    ).reshape(x.shape)

    logp = torch.masked_fill(logp, ~mask, 0.0)
    logp_seq = logp.sum(-1) / (mask).float().sum(-1)

    return torch.exp(-logp_seq).mean()


@torch.no_grad()
def acc(logits, x, tokenizer):
    x, logits = x[:, 1:], logits[:, :-1]
    mask = x != tokenizer.pad_token_id

    corr = torch.logical_and(
        logits.argmax(-1) == x,
        mask
    ).float().sum()
    return corr / (mask).float().sum()


@torch.no_grad()
def pcorr(logits, x, tokenizer):
    x = x[:, 1:].contiguous().view(-1)
    logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
    mask = x != tokenizer.pad_token_id

    logp = -F.cross_entropy(
        logits, x,
        reduction='none'
    )
    p = torch.exp(logp)

    p = torch.masked_fill(p, ~mask, 0.0)
    return p.sum() / (mask).float().sum()
