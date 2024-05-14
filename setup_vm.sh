#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token
    $2: wandb token

Example:
    . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
'

# upgrade pip to get higher torch_xla version
python -m pip install pip --upgrade

# install torch stuff
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# update path
export PATH="/home/$USER/.local/bin:$PATH"

# install extras
pip install transformers datasets webdataset wandb matplotlib

# install jax for flash attention
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# login to huggingface
huggingface-cli login --token $1 --add-to-git-credential

# login to wandb
python -m wandb login $2
