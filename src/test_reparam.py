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

    phi.grad = None

    # get reparam loss
    loss = 1000*phi[0]
    z = (torch.exp(logp_lm + phi)).sum().detach()

    print(torch.softmax(logp_lm, dim=-1))
    counts = torch.zeros_like(logp_lm)
    for _ in range(1000):
        sample = torch.multinomial(torch.exp(logp), 1).item()

        reparam = z.detach() + torch.exp(phi[sample]) - torch.exp(phi[sample]).detach()
        loss = loss - torch.log(reparam)

        counts[sample] += 1
    print(counts/1000)
    
    loss = -loss/1000
    loss.backward()

    print(phi.grad)


if __name__ == '__main__':
    main()