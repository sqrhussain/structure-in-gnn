*!!! DEPRECATED !!!* - Please refer to (https://github.com/sqrhussain/homophily-community-gnn)

# Network structure impact on graph neural networks


Dependencies: to be added

### Papers
This code is the basis for our

* [On the Impact of Communities on Semi-supervised Classification Using Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-65351-4_2)
* The Interplay between Communities and Homophily in Semi-supervised Classification Using Graph Neural Networks - [a preprint is currently available](https://sqrhussain.github.io/papers/apns21.pdf)

### Citation
If you use our code/paper please cite our work :)
```
@inproceedings{hussain2020impact,
  title={On the Impact of Communities on Semi-supervised Classification Using Graph Neural Networks},
  author={Hussain, Hussain and Duricic, Tomislav and Lex, Elisabeth and Kern, Roman and Helic, Denis},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={15--26},
  year={2020},
  organization={Springer}
}
```

### Datasets
First, run `python -m src.data.dataset_handle` to download and transform the datasets. This works for Cora, Citeseer, WebKB and Pubmed without hassle. Needs some tweaks to work on other datasets (to be fixed/explained).

### Generating synthetic graphs from the original ones
#### Configuration model
Eliminates community structure while keeping the degree sequence.

Use `python -m src.data.create_configuration_model`

#### Stochastic block model
Eliminates the skew in the degree distribution (approaches a binomial distribution) while aiming to preserve the community structure using Louvain method for community detection.

Use `python -m src.data.create_configuration_model`

#### Erdős–Rényi model
Eliminates the community structure and turns the degree distribution into a binomial distribution. The only preserved properties are the node sequence (number, identity and features) and an approximate edge density.

Use `python -m src.data.create_random_graph`

### Hyperparameter optimization
Run `python -m src.hyperparam_search` with the suitable parameters. You can modify the ranges within the python file. Stores validation results in `reports/results/eval/` which will be necessary to run `train.py`.

### Training and evaluating
Run `python -m src.train` with the suitable parameters. This file uses the resutls from the previous step and stores the evaluation resutls in `reports/results/test_acc`.

### Computing the uncertainty coefficient
View the `notebooks/Uncertainty coefficient.ipynb` for details

