##  Architecture Search

To search a GNN model, you can run the script `run_pspnas.sh`, the parameters of which is listed in order:

+ **dataset** ( Cora / Citeseer / Pubmed / chameleon / cornell / texas / wisconsin )

+ **use_early_stop** (0: use;  1: not use )

+ **split_type** ( standard / full_supervised )

  > **Note:** Standard split type is only used for Cora, Citeseer and Pubmed. 

### Run Script - Example 01

To search the best architecture as well as its hyperparamaters on Cora with standard split, please run: 

```
bash run_pspnas.sh Cora 0 standard
```

### Run Script - Example 02

To search the best architecture as well as its hyperparamaters on chameleon with full_supervised split, please run: 

```
bash run_pspnas.sh chameleon 0 full_supervised
```

### Code Running Log

The log result of running searching code will be saved to the directory `log_output/` as a file's name in form of `dataset_name-%Y-%m-%d-%H-%M-%S.log`

