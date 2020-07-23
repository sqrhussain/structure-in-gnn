# Network structure impact on graph neural networks


Dependencies: to be added

### Datasets
Run `python -m src.data.dataset_handle`. This works for Cora, Citeseer, WebKB and Pubmed without hassle. Needs some tweaks to work on other datasets (to be fixed/explained).

### Generating synthetic graphs from the original ones
#### Configuration model
Eliminates community structure while keeping the degree sequence.
Use `python -m src.data.create_configuration_model`
#### Stochastic block model
Eliminates the skew in the degree distribution (approaches a binomial distribution) while aiming to preserve the community structure using Louvain method for community detection.
Use `python -m src.data.create_configuration_model`
#### Erdős–Rényi model
Eliminates the community structure and turns the degree distribution into a binomial distribution. The only preserved properties are the node sequence (number, identity and features) and an approximate edge density.

### Hyperparameter optimization
Run `python -m src.hyperparam_search` with the suitable parameters. You can modify the ranges within the python file. Stores validation results in `reports/results/eval/` which will be necessary to run `train.py`.

### Training and evaluating
Run `python -m src.train` with the suitable parameters. This file uses the resutls from the previous step and stores the evaluation resutls in `reports/results/test_acc`.

