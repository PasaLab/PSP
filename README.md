### PSP: Progressive Space Pruning for Efficient Graph Neural Architecture Search

#### Requirements

Ensure that python version >= 3.6, pytorch version >= 1.4.0 . Then run:

```
pip install -r requirements.txt
```

+ Notes: This requirements file include torch-geometric == 1.6.3 (Two key componets)

#### Datasets

#### Architecture Search

To run the search process, please refer to `offline/psp_nas/run_pspnas_offline.sh`for PSP-NAS. 

For example, search a 2 nodes 2cells GNN model on Cora dataset with standard split, please run:

```
cd offline/psp_nas
# parameters explanation respectively: dataset_name, is_use_early_stop(default 0 :not use), split_type
sh run_pspnas_offline.sh Cora 0 standard
```

To search the GNN model on Cora dataset with full_supervised split, please run:

```
cd offline/psp_nas
sh run_pspnas_offline.sh Cora 0 full_supervised
```

And notes that, there is only full_supervised split to Chameleon, Cornell, Texas and Wisconsin.

