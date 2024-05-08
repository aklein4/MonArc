: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token

Example:
    . setup.sh <Huggingface token>
'

# upgrade pip to get higher torch_xla version
python -m pip install pip --upgrade

# install torch stuff
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# update path
export PATH="/home/$USER/.local/bin:$PATH"

# install extras
pip install transformers datasets webdataset matplotlib

# set to use TPU
export PJRT_DEVICE=TPU

# login to huggingface
huggingface-cli login --token $1 --add-to-git-credential
