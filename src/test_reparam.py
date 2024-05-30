import torch


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
    print(phi.grad)

    prev_grad = phi.grad.detach().clone()
    phi.grad = None

    # get reparam loss
    grad_accum = torch.zeros_like(phi)
    counts = torch.zeros_like(phi)
    for _ in range(10000):

        loss = phi[0]
        z = torch.exp(logp_lm + phi).sum().detach()

        sample = torch.multinomial(torch.softmax(logp_lm, dim=-1), 1).item()
        loss = loss - (1/z) * torch.exp(phi[sample]) / torch.exp(logp_lm).sum()
        # loss = loss - (1/z) * (torch.exp(logp_lm) * torch.exp(phi)).sum()

        loss = -loss
        loss.backward()

        grad_accum += phi.grad
        phi.grad = None

        counts[sample] += 1
    
    # loss = phi[0]
    # # print(torch.softmax(logp_lm, dim=-1))
    # # print(counts/10000)
    # loss = loss - (1/z) * ((counts/10000) * torch.exp(phi)).mean()

    # loss = -loss
    # loss.backward()

    # print(phi.grad)
    # return

    print(grad_accum/10000)
    print(((grad_accum/10000) / prev_grad))
    # print(torch.softmax(logp_lm, dim=-1))
    # print(torch.softmax(phi, dim=-1))
    # print(torch.softmax(logp_lm + phi, dim=-1))


if __name__ == '__main__':
    main()