import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(logits, x, ignore_index):
    x, logits = x[:, 1:], logits[:, :-1]

    return F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        ignore_index=ignore_index
    )


@torch.no_grad()
def ppl(logits, x, ignore_index):
    x = x[:, 1:]
    logits = logits[:, :-1]
    mask = x != ignore_index

    logp = -F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        reduction='none'
    ).view(x.shape)

    logp = torch.masked_fill(logp, ~mask, 0.0)
    logp_seq = logp.sum(-1) / (mask).float().sum(-1)

    return torch.exp(-logp_seq).mean()


@torch.no_grad()
def acc(logits, x, ignore_index):
    x, logits = x[:, 1:], logits[:, :-1]
    mask = x != ignore_index

    corr = torch.logical_and(
        logits.argmax(-1) == x,
        mask
    ).float().sum()
    return corr / (mask).float().sum()


@torch.no_grad()
def pcorr(logits, x, ignore_index):
    x = x[:, 1:].contiguous().view(-1)
    logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
    mask = x != ignore_index

    logp = -F.cross_entropy(
        logits, x,
        reduction='none'
    )
    p = torch.exp(logp)

    p = torch.masked_fill(p, ~mask, 0.0)
    return p.sum() / (mask).float().sum()


def arc_loss(true_arc, fake_arc, input_ids, ignore_index, divisor=1):
    assert input_ids.shape[0] % divisor == 0
    div_bs = input_ids.shape[0] // divisor

    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    true_input_ids = input_ids[:, 1:].view(-1)
    fake_input_ids = input_ids[:div_bs, 1:].view(-1)

    true_loss = F.logsigmoid(-true_arc)
    fake_loss = F.logsigmoid(fake_arc)

    true_mask = true_input_ids != ignore_index
    fake_mask = fake_input_ids != ignore_index
    true_loss = torch.masked_fill(true_loss, ~true_mask, 0.0)
    fake_loss = torch.masked_fill(fake_loss, ~fake_mask, 0.0)

    return -(
        true_loss.sum()/true_mask.float().sum() +
        fake_loss.sum()/fake_mask.float().sum()
    )


def arc_acc(true_arc, fake_arc, input_ids, ignore_index, divisor=1):
    assert input_ids.shape[0] % divisor == 0
    div_bs = input_ids.shape[0] // divisor
    
    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    true_input_ids = input_ids[:, 1:].view(-1)
    fake_input_ids = input_ids[:div_bs, 1:].view(-1)
    
    true_acc = (true_arc < 0).float()
    fake_acc = (fake_arc >= 0).float()

    true_mask = true_input_ids != ignore_index
    fake_mask = fake_input_ids != ignore_index
    true_acc = torch.masked_fill(true_acc, ~true_mask, 0.0)
    fake_acc = torch.masked_fill(fake_acc, ~fake_mask, 0.0)

    return 0.5 * (
        true_acc.sum()/true_mask.float().sum() +
        fake_acc.sum()/fake_mask.float().sum()
    )


def arc_pcorr(true_arc, fake_arc, input_ids, ignore_index, divisor=1):
    assert input_ids.shape[0] % divisor == 0
    div_bs = input_ids.shape[0] // divisor

    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    true_input_ids = input_ids[:, 1:].view(-1)
    fake_input_ids = input_ids[:div_bs, 1:].view(-1)

    true_p = torch.sigmoid(-true_arc)
    fake_p = torch.sigmoid(fake_arc)

    true_mask = true_input_ids != ignore_index
    fake_mask = fake_input_ids != ignore_index
    true_p = torch.masked_fill(true_p, ~true_mask, 0.0)
    fake_p = torch.masked_fill(fake_p, ~fake_mask, 0.0)

    return 0.5 * (
        true_p.sum()/true_mask.float().sum() +
        fake_p.sum()/fake_mask.float().sum()
    )
