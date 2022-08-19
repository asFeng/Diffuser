# Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long Sequences

Diffuser combines advantages of both the efficiency of sparse transformer and the expressiveness of full-attention Transformer.
The key idea is to expand the receptive field of sparse attention using Attention Diffusion,
which computes multi-hop token correlations based on all paths between corresponding disconnected tokens, besides attention among neighboring tokens.

## Code
Important files in the code are:
*   [diffuser_att.py](./models/diffuser_att.py): The core attention module
*   [diffuser.py](./models/diffuser.py): The Diffuser layer architecture
*   [diffuser_app.py](./models/diffuser_app.py): Application wrappers for different tasks
*   [graphtrainer.py](./graphtrainer.py): Customized trainer defining sparse patterns as graphs

### Installation 
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.8.0 version.

The other important dependencies requirements are listed in [requirements.txt](./requirements.txt).

### Running Classification
To run IMDB review classification task with one GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train_classification_imdb.py 
```
Multi-GPU training has to be lauched with DistributedDataParallel (DDP) for PyTorch and DGL
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_classification_imdb.py
 ```
Model configurations are listed in [config.json](./models/config.json) and training arguments can be changed in [train_classification_imdb.py](./train_classification_imdb.py)