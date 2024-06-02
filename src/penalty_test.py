import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt


LR = 1e-3
W_PEN = 1000.0
FORCE = 0.0


def main():
    
    targ = nn.Parameter(torch.zeros(1))
    est = nn.Parameter(torch.ones(1))

    optimizer = torch.optim.AdamW([targ, est], lr=LR)

    targs = [targ.item()]
    ests = [est.item()]
    for _ in tqdm(range(10000)):

        loss = (targ.detach() - est).pow(2)
        loss = loss + FORCE*targ

        est_reparam = est.detach() + targ - targ.detach()
        loss = loss + W_PEN*est_reparam.pow(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targs.append(targ.item())
        ests.append(est.item())
    
    plt.plot(targs, label="targ")
    plt.plot(ests, label="est")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()