# Monarc

### Overview

MonArc is a practical method to train energy-based language models (ELMs) using a residual energy framework. In this framework, we have an autoregressive language model (LM) that samples candidates for generation, and a residual energy-based model (EBM) that resamples from those candidates to improve accuracy.

Previous residual energy methods require 'negative' sequences to be sampled from the LM at training time, which is a computational bottleneck due to the non-parallelizable nature of autoregressive sampling. MonArc overcomes this limitation by having the residual EBM operate on the token level, rather than the sequence level. This means that sampling negatives only requires one parallelizable pass through the LM to generate negatives, greatly improving efficiency.

When using a single causal transformer decoder as both the LM and residual EBM, MonArc has shown improved performance over LM baselines in terms of both data and compute at training time. In the results below, we see that using MonArc to fine-tune an existing LM quickly reduces the loss of the full ELM, while maintaining its standard LM capabilities.

### Results

...

## TPU VM Setup Instructions

1. Create VM with version: `tpu-ubuntu2204-base`

2. `git clone https://github.com/aklein4/QLM.git`

3. `cd ~/QLM && . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>`
