import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    # init variables
    logp_lm = torch.randn(16)
    phi = torch.randn(16)

    logp_lm.requires_grad = False
    phi.requires_grad = True

    # get true loss
    logp = torch.log_softmax(logp_lm+phi, dim=-1)
    true_loss = -logp[0]
    true_loss.backward()

    prev_grad = phi.grad.detach().clone()
    phi.grad = None

    # get reparam loss
    z = (torch.softmax(logp_lm, dim=-1) * torch.exp(phi)).sum().detach()
    grad_accum = torch.zeros_like(phi)
    for _ in range(10000):

        loss = phi[0]

        sample = torch.multinomial(torch.softmax(logp_lm, dim=-1), 1).item()
        loss = loss - (1/z) * torch.exp(phi[sample])

        loss = -loss
        loss.backward()

        grad_accum += phi.grad
        phi.grad = None

    print(prev_grad)
    print(grad_accum/10000)
    print(((grad_accum/10000) / prev_grad))


def prototype():
    
    logp_targ = torch.randn(16)
    logp_lm = torch.randn(16)

    logp_targ = torch.sort(logp_targ, descending=True)[0]

    z_est = nn.Parameter(torch.ones(1))
    phi = nn.Parameter(torch.zeros(16))

    optimizer = torch.optim.Adam([z_est, phi], lr=1e-3)

    for _ in tqdm(range(10000)):

        targ_sample = torch.multinomial(torch.softmax(logp_targ, dim=-1), 1).item()
        lm_sample = torch.multinomial(torch.softmax(logp_lm, dim=-1), 1).item()

        loss = -(phi[targ_sample] - torch.exp(phi[lm_sample]) / z_est.detach())
        # loss = -(torch.log_softmax(logp_lm + phi, dim=-1)[targ_sample])

        loss = loss + (z_est - torch.exp(phi[lm_sample]).detach()) ** 2

        z_reparam = z_est.detach() + torch.exp(phi[lm_sample]) - torch.exp(phi[lm_sample]).detach()
        loss = loss + (1/z_reparam - 1) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Error:")
    print(-F.kl_div(logp_targ[None], (logp_lm+phi)[None], log_target=True, reduction='batchmean').item())
    print("Target:")
    print(F.softmax(logp_targ, dim=-1))
    print("Emulation:")
    print(F.softmax(logp_lm + phi, dim=-1))
    print("True Z:")
    print((torch.softmax(logp_lm, dim=-1) * torch.exp(phi)).sum().item())
    print("Estimated Z:")
    print(z_est.item())

    plt.plot(F.softmax(logp_targ, dim=-1).detach().numpy(), label="Target")
    plt.plot(F.softmax(logp_lm + phi, dim=-1).detach().numpy(), label="Emulation")
    plt.legend()
    plt.show()


def cross_entropy_loss(logp, lm, phi):
    loss = -torch.softmax(logp, -1)*(
        torch.log_softmax(lm - phi, dim=-1)
    )
    return loss.mean()


def arc_loss(logp, lm, phi):    
    l1 = -torch.softmax(logp, -1)*(
        -phi
    )
    l2 = -torch.softmax(lm, -1)*(
        torch.exp(-phi).detach() * (-phi)
    )
    return (l1 + l2).mean()


def cne_loss(logp, lm, phi):
    l1 = -torch.softmax(logp, -1)*(
        F.logsigmoid(-phi)
    )
    l2 = -torch.softmax(lm, -1)*(
        F.logsigmoid(phi)
    )
    return (l1 + l2).mean()


def comparison():

    logp_targ = torch.randn(64)
    logp_targ = torch.sort(logp_targ, descending=True)[0]
    logp_lm = logp_targ + torch.randn(64)

    logp_targ = F.log_softmax(logp_targ, dim=-1)
    logp_lm = F.log_softmax(logp_lm, dim=-1)

    kl_dict = {}
    z_dict = {}
    for name, loss_fn in [
        ("cross_entropy", cross_entropy_loss),
        ("arc", arc_loss),
        ("cne", cne_loss)
    ]:

        phi = nn.Parameter(torch.zeros(64))

        optimizer = torch.optim.Adam([phi], lr=1e-3)

        kls = []
        zs = []
        for _ in tqdm(range(5000)):

            loss = loss_fn(logp_targ, logp_lm, phi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kl = (torch.softmax(logp_targ, dim=-1) * (
                    torch.log_softmax(logp_targ, dim=-1) -
                    torch.log_softmax(logp_lm-phi, dim=-1)
                )).sum().item()
                kls.append(kl)

                zs.append(torch.log((torch.softmax(logp_lm, dim=-1) * torch.exp(-phi)).sum()).item())

        kl_dict[name] = kls
        z_dict[name] = zs
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    colors = {
        "cross_entropy": "blue",
        "arc": "black",
        "cne": "red"
    }
    for name, kls in kl_dict.items():
        ax[0].plot(kls, label=name, color=colors[name])
        ax[1].plot(z_dict[name], label=name, color=colors[name])

    ax[0].legend()
    ax[0].set_title("KL Divergence")

    ax[1].legend()
    ax[1].set_title("Z")

    plt.show()


def penalty_test():
    
    logz = 3*torch.randn(1)
    phi = nn.Parameter(torch.randn(1))
    logz += phi.detach()

    logz_reparam = torch.log(
        logz.exp().detach() +
        torch.exp(-phi) -
        torch.exp(-phi).detach()
    )
    base_loss = logz_reparam.pow(2)
    base_loss.backward()

    print("Base Grad:", phi.grad)   
    phi.grad = None

    logz_reparam = (
        logz.detach() +
        torch.exp(-phi - logz).detach() * ((-phi) - (-phi).detach())
    )
    reparam_loss = logz_reparam.pow(2)
    reparam_loss.backward()

    print("Reparam Grad:", phi.grad)
    

if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # main()
    # prototype()
    comparison()
    # penalty_test()