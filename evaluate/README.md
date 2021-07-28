## Evaluate

### Architecture Structure

For Cora with standard split, architecture cell searched by PSP_NAS together with its hyperparameters, e.g. `[1, 'cos', 'tanh', 2, 'appnp', 'elu', 'add'], lr=5e-4, dropout=0.8, weight_decay_gnn=1e-3, weight_decay_fc=5e-3, hidden_num=512`, and the corressponding cell visualization as follows:

![Cora_standard](https://github.com/PasaLab/PSP/blob/master/figures/Cora_standard.png)

### Run Script

To evaluate architectures searched by PSP_NAS or by their own, you can run the script `run_evaluate_single_model.sh`, the parameters are listed as `dataset, split_type`.

#### Run Script for Single Architecture

To evaluate the best architecture searched by PSP_NAS  as well as its hyperparamaters on Cora with standard split, please run: 

```
bash run_evaluate_single_model.sh Cora standard
```

#### Run Script for Architecute for 10 Splits in Full_supervised Split Type

To evaluate the best architecture searched by PSP_NAS  as well as its hyperparamaters on Chameleon with full_supervised in 10 splits, please run: 

```
bash run_evaluate_10splits.sh chameleon full_supervised
```

### Code Running Log

The log result of running evaluating code will be saved to the directory `evaluate_log_output/` as a file's name in form of `dataset_name-%Y-%m-%d-%H-%M-%S.log`

### Results

> Node classification task for Cora, Citeseer and Pubmed w.r.t. accuracy

|               | Cora                |                     | Citeseer                |                       | Pubmed                     |                       |
| :-----------: | ------------------- | ------------------- | ----------------------- | --------------------- | -------------------------- | --------------------- |
|               | standard            | Full                | standard                | Full                  | standard                   | full                  |
|    **GCN**    | 81.5                | 85.77               | 71.1                    | 73.68                 | 79.0                       | 88.13                 |
|    **GAT**    | 83.1                | 86.37               | 71.9                    | 74.32                 | 78.5                       | 87.62                 |
|   **APPNP**   | 83.3                | <u>87.87</u>        | 71.8                    | <u>76.53</u>          | 80.1                       | <u>89.40</u>          |
|   **ARMA**    | 83.4                | /                   | 72.5                    | /                     | 78.9                       | /                     |
|   **JKNet**   | 81.1(4)             | 75.85(8)            | 69.8(16)                | 75.85(8)              | 78.1(32)                   | 88.94(64)             |
|   **GCNII**   | **84.2**            | 86.12               | 70.6                    | 75.80                 | <u>80.3</u>                | 79.3                  |
|               |                     |                     |                         |                       |                            |                       |
| **RandomNAS** | 83.16 +/- 0.5       | 87.35 +/- 1.21      | 72.53 +/- 0.23          | <u>76.75 +/- 1.73</u> | 78.21 +/- 0.34             | <u>89.96 +/- 0.40</u> |
|  **AutoGNN**  | <u>83.6 +/- 0.3</u> | /                   | <u>**73.8 +/- 0.7**</u> | /                     | <u>79.7 +/- 0.4</u>        | /                     |
|   **SNAG**    | 77.8 +/- 0.95       | 82.12 +/- 1.04      | 61.5 +/- 1.04           | 70.66 +/- 1.37        | 72.08 +/- 1.09             | 85.48 +/- 0.39        |
| **GraphNAS**  | 82.10 +/- 0.77      | 87.06 +/- 0.95      | 72.00 +/- 0.35          | 76.25 +/- 1.90        | 79.01 +/- 0.48             | 89.70 +- 0.48         |
|               |                     |                     |                         |                       |                            |                       |
|   **Ours**    | 83.84 +/- 0.40      | **88.17 +/- 1.459** | 73.35 +/- 0.36          | **77.13 +/- 1.86**    | <u>**80.53 +/- 0.320**</u> | **90.04 +/- 0.44**    |



> Node classification task for four datasets with low node homophily w.r.t accuracy

|                | chamelon              | cornell               | texas                 | wisconsin             |
| :------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| **GCN**        | 28.18                 | 52.70                 | 52.16                 | 45.88                 |
| **GAT**        | 42.93                 | 54.32                 | 58.38                 | 49.41                 |
| **APPNP **     | 54.3                  | 55.68                 | 65.41                 | 69.02                 |
| **JKNet**      | 60.07                 | 57.30                 | 56.49                 | 48.82                 |
| **GCNII **     | <u>61.14</u>          | <u>77.03</u>          | <u>76.76</u>          | <u>82.16</u>          |
| **Geom-GCN-I** | 60.31                 | 56.76                 | 57.58                 | 58.24                 |
| **Geom-GCN-p** | 60.90                 | 60.81                 | 67.57                 | 64.12                 |
| **Geom-GCN-S** | 59.96                 | 55.68                 | 59.73                 | 56.67                 |
|                |                       |                       |                       |                       |
| **RandomNAS**  | <u>66.55 +/- 1.57</u> | 79.57 +/- 6.02        | 79.40 +/- 4.20        | 84.44 +/- 2.77        |
| **SNAG**       | 21.66 +/- 0.67        | 24 +/- 1.77           | 60.81 +/- 4.90        | 45.06 +/- 5.90        |
| **GraphNAS**   | 66.11 +/- 2.70        | <u>81.89 +/- 6.34</u> | <u>80.36 +/- 5.65</u> | <u>84.62 +/- 2.28</u> |
|                |                       |                       |                       |                       |
| **Ours**       | **69.37 +/- 1.94**    | **83.20 +/- 5.53**    | **82.06 +/- 7.82**    | **85.70 +/- 2.83**    |

