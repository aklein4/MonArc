from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import log_master_print


def loss(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Standard cross-entropy loss for language modeling.
     - applies offset so that logits_{t} predicts x_{t+1}
     - ignores padding tokens and last logits
     
    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: cross-entropy loss [nats]
    """
    x, logits = x[:, 1:], logits[:, :-1]

    return F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        ignore_index=ignore_index
    )


@torch.no_grad()
def ppl(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute perplexity of the model.
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: Perplexity [nats]
    """
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
def acc(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute top-1 next-token accuracy of the model.
     - uses same data logic as loss()
    
    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: top-1 token accuracy
    """
    x, logits = x[:, 1:], logits[:, :-1]
    mask = x != ignore_index

    corr = torch.logical_and(
        logits.argmax(-1) == x,
        mask
    ).float().sum()
    return corr / (mask).float().sum()


@torch.no_grad()
def pcorr(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute token prediction probability of the model.
     - measures probability that a token sampled from logits is equal to target token
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: next-token prediction probability
    """
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


def arc_loss(
    true_arc: torch.Tensor,
    fake_arc: torch.Tensor,
    input_ids: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute the loss for arc models.
     - uses contrastive noise estimation loss
     - negative arc = more likely to be true example
     - arc_{t} is ignored if input_ids_{t+1} is padding token
     - last item in sequence is ignored

    Args:
        true_arc (torch.Tensor): arc values for true examples [B, T, 1]
        fake_arc (torch.Tensor): arc values for fake examples [B, T, 1]
        input_ids (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: noise contrastive loss [nats]
    """
    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    true_loss = F.logsigmoid(-true_arc)
    fake_loss = F.logsigmoid(fake_arc)

    mask = input_ids != ignore_index
    true_loss = torch.masked_fill(true_loss, ~mask, 0.0)
    fake_loss = torch.masked_fill(fake_loss, ~mask, 0.0)

    return -(
        true_loss.sum() +
        fake_loss.sum()
    )/mask.float().sum()


@torch.no_grad()
def arc_acc(
    true_arc: torch.Tensor,
    fake_arc: torch.Tensor,
    input_ids: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute the arc prediction accuracy.
     - accurate if true_arc < 0 and fake_arc >= 0
     - uses same data logic as arc_loss()

    Args:
        true_arc (torch.Tensor): arc values for true examples [B, T, 1]
        fake_arc (torch.Tensor): arc values for fake examples [B, T, 1]
        input_ids (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: arc descriminator accuracy
    """
    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)
    
    true_acc = (true_arc < 0).float()
    fake_acc = (fake_arc >= 0).float()

    mask = input_ids != ignore_index
    true_acc = torch.masked_fill(true_acc, ~mask, 0.0)
    fake_acc = torch.masked_fill(fake_acc, ~mask, 0.0)

    return 0.5 * (
        true_acc.sum() +
        fake_acc.sum()
    )/mask.float().sum()


@torch.no_grad()
def arc_pcorr(
    true_arc: torch.Tensor,
    fake_arc: torch.Tensor,
    input_ids: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Compute the correct arc prediction probability.
     - measures probability that a true/false value sampled from descriminator is correct
     - uses same data logic as arc_loss()

    Args:
        true_arc (torch.Tensor): arc values for true examples [B, T, 1]
        fake_arc (torch.Tensor): arc values for fake examples [B, T, 1]
        input_ids (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: arc descriminator prediction probability
    """
    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    true_p = torch.sigmoid(-true_arc)
    fake_p = torch.sigmoid(fake_arc)

    mask = input_ids != ignore_index
    true_p = torch.masked_fill(true_p, ~mask, 0.0)
    fake_p = torch.masked_fill(fake_p, ~mask, 0.0)

    return 0.5 * (
        true_p.sum() + 
        fake_p.sum()
    )/mask.float().sum()


@torch.no_grad()
def arc_adj(
    true_arc: torch.Tensor,
    fake_arc: torch.Tensor,
    input_ids: torch.LongTensor,
    ignore_index: Optional[int]=-1
) -> torch.Tensor:
    """ Estimate of the average log probability gain when z=1, in nats. (higher better)
     - only accurate in ideal z=1 case, which occurs with a perfect descriminator
     - uses same data logic as arc_loss()

    Args:
        true_arc (torch.Tensor): arc values for true examples [B, T, 1]
        fake_arc (torch.Tensor): arc values for fake examples [B, T, 1]
        input_ids (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: log probability gain when z=1 (higher  better)
    """
    true_arc = true_arc[:, :-1].view(-1)
    fake_arc = fake_arc[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    # appriximate log prob gain when z=1
    adj = -true_arc

    mask = input_ids != ignore_index
    adj = torch.masked_fill(adj, ~mask, 0.0)

    return adj.sum()/mask.float().sum()


def reaper_phi_loss(
    true_res,
    fake_res,
    logz,
    input_ids,
    ignore_index=-1
):
    true_res = true_res[:, :-1].view(-1)
    fake_res = fake_res[:, :-1].view(-1)
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    loss = (
        (-true_res) -
        torch.exp(-fake_res - logz).detach() * (-fake_res)
    )

    mask = input_ids != ignore_index
    loss = torch.masked_fill(loss, ~mask, 0.0)

    return -loss.sum()/mask.float().sum()


def reaper_z_loss(
    fake_res,
    logz_dist,
    input_ids,
    ignore_index=-1,
):
    fake_res = fake_res[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    logp = logz_dist.log_prob(-fake_res.detach(), remove_last=True)
    loss = -logp

    mask = input_ids != ignore_index
    loss = torch.masked_fill(loss, ~mask, 0.0)

    return loss.sum()/mask.float().sum()


def reaper_penalty(
    fake_res,
    logz,
    input_ids,
    ignore_index=-1
):
    fake_res = fake_res[:, :-1].view(-1)
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    logz_reparam = (
        logz.detach() +
        torch.exp(-fake_res - logz).detach() * (
            (-fake_res) - (-fake_res).detach()
        )
    )
    loss = logz_reparam.pow(2)

    mask = input_ids != ignore_index
    loss = torch.masked_fill(loss, ~mask, 0.0)

    return loss.sum()/mask.float().sum()


@torch.no_grad()
def reaper_adj(
    true_res,
    logz,
    input_ids,
    ignore_index=-1
):
    true_res = true_res[:, :-1].view(-1)
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    adj = -true_res - logz

    mask = input_ids != ignore_index
    adj = torch.masked_fill(adj, ~mask, 0.0)

    return adj.sum()/mask.float().sum()


@torch.no_grad()
def reaper_check(
    fake_res,
    logz,
    input_ids,
    ignore_index=-1
):
    fake_res = fake_res[:, :-1].view(-1)
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    check = torch.exp(-fake_res - logz)

    mask = input_ids != ignore_index
    check = torch.masked_fill(check, ~mask, 0.0)

    return check.sum()/mask.float().sum()


@torch.no_grad()
def reaper_sample_abs(
    fake_res,
    input_ids,
    ignore_index=-1
):
    fake_res = fake_res[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    out = fake_res.abs()

    mask = input_ids != ignore_index
    out = torch.masked_fill(out, ~mask, 0.0)

    return out.sum()/mask.float().sum()


@torch.no_grad()
def reaper_logz_abs(
    logz,
    input_ids,
    ignore_index=-1
):
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    out = logz.abs()

    mask = input_ids != ignore_index
    out = torch.masked_fill(out, ~mask, 0.0)

    return out.sum()/mask.float().sum()


@torch.no_grad()
def reaper_sample(
    fake_res,
    input_ids,
    ignore_index=-1
):
    fake_res = fake_res[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    out = -fake_res.clone()

    mask = input_ids != ignore_index
    out = torch.masked_fill(out, ~mask, 0.0)

    return out.sum()/mask.float().sum()


@torch.no_grad()
def reaper_true(
    true_res,
    input_ids,
    ignore_index=-1
):
    true_res = true_res[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    out = -true_res.clone()

    mask = input_ids != ignore_index
    out = torch.masked_fill(out, ~mask, 0.0)

    return out.sum()/mask.float().sum()


@torch.no_grad()
def reaper_logz(
    logz,
    input_ids,
    ignore_index=-1
):
    logz = logz[:, :-1].view(-1)
    input_ids = input_ids[:, 1:].view(-1)

    out = logz.clone()

    mask = input_ids != ignore_index
    out = torch.masked_fill(out, ~mask, 0.0)

    return out.sum()/mask.float().sum()