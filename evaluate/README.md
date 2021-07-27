## Evaluate

### Architecture Structure

Architecture cell searched by PSP_NAS together with its hyperparameters, e.g. `[1, 'cos', 'tanh', 2, 'appnp', 'elu', 'add'], lr=5e-4, dropout=0.8, weight_decay_gnn=1e-3, weight_decay_fc=5e-3, hidden_num=512`, and the corressponding cell visualization as follows:

![Cora_standard](/Users/wwj/workspace/ipynb/visualize/Cora_standard.png)

### Run Script

To evaluate architectures searched by PSP_NAS or by their own, you can run the script `run_evaluate_single_model.sh`, the parameters are listed as `dataset, split_type`.

#### Run Script for Single Architecture

To evaluate the best architecture searched by PSP_NAS  as well as its hyperparamaters on Cora with standard split, please run: 

```python
bash run_evaluate_single_model.sh Cora standard
```

#### Run Script for Architecute for 10 Splits in Full_supervised Split Type

To evaluate the best architecture searched by PSP_NAS  as well as its hyperparamaters on Chameleon with full_supervised in 10 splits, please run: 

```python
bash run_evaluate_10splits.sh chameleon full_supervised
```

### Code Running Log

The log result of running evaluating code will be saved to the directory `evaluate_log_output/` as a file's name in form of `dataset_name-%Y-%m-%d-%H-%M-%S.log`

