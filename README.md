# QLM
quasi-causal language models


## VM Start Instructions

1. Create VM with version: `tpu-ubuntu2204-base`

2. `python -m pip install pip --upgrade`

3. `pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html`

4. `export PATH="/home/$USER/.local/bin:$PATH"` (make sure to replace USER)

5. `pip install transformers datasets webdataset matplotlib`

6. `export PJRT_DEVICE=TPU`

7. `huggingface-cli login`

8. `git clone https://github.com/aklein4/QLM.git`