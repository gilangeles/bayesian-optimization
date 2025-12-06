# Hyperparameter tuning by Bayesian Optimization

## How to get started

1. install `uv`
2. run `uv run --extra cu128 python main.py train` to init the virtual environment and run the training for GPU-enabled machines and `uv run --extra cpu python main.py train` for CPU-only machines
3. run `uv run python main.py train --help` for more help on arguments

`uv run` automatically creates a virtual environment that installs all the necessary requirements needed to run the script.


## Folder structure

```
.
|-- README.md
|-- bayesian_optimizer
|   |-- __init__.py
|   |-- acquisition.py
|   `-- plot_graph.py
|-- main.py
|-- model
|   |-- __init__.py
|   `-- resnet.py
|-- pyproject.toml
`-- trainer
    |-- __init__.py
    |-- callbacks.py
    |-- common.py
    `-- lit_resnet.py

```


## Developing

- install `pre-commit` using `pip install pre-commit`. Install hook by doing a `pre-commit install`


## References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv.
2. [Self-Adjusting Weighted Expected Improvement for Bayesian Optimization](https://arxiv.org/pdf/230604262)Benjamins, C., Raponi, E., Jankovic, A., Doerr, C., & Lindauer, M. (2023). Self-Adjusting Weighted Expected Improvement for Bayesian Optimization. arXiv.
3. [Towards Assessing the Impact of Bayesian Optimization’s Own Hyperparameters](https://arxiv.org/pdf/1908.06674)Lindauer, M., Feurer, M., Eggensperger, K., Biedenkapp, A., & Hutter, F. (2019). Towards assessing the impact of Bayesian Optimization’s own hyperparameters. DSO Workshop at IJCAI. arXiv.
4. [PyTorch Lightning](https://www.pytorchlightning.ai) Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935
5. [Bayesian Machine Learning](https://github.com/krasserm/bayesian-machine-learning)
