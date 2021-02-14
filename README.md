# Contrastive Poisson latent variable models

This repo contains models and algorithms for contrastive Poisson latent variable models (CPLVM). Given a count-based foreground dataset and a count-based backround dataset, the CPLVM is designed to find structure and variation that is enriched in the foreground relative to the background.


The accompanying paper can be found here: XXX.

## Installation

After cloning this repo, navigate to its directory, and run the following command.
```
git clone git@github.com:andrewcharlesjones/cplvm.git
cd cplvm
python setup.py install
```

You should then be able to import the model as follows:
```python
from cplvm import CPLVM
```
## Example

