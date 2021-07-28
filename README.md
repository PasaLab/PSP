## PSP: Progressive Space Pruning for Efficient Graph Neural Architecture Search

### Abstract

​	Recently, graph neural network (GNN) has achieved great success in many graph learning tasks such as node classification and graph classification. However, there is no single GNN architecture that can fit different graph datasets. Designing an effective GNN for a specific graph dataset requires considerable expert experience and huge computational costs. Inspired by the success of neural architecture search (NAS), searching the GNN architectures automatically has attracted more and more attention. Motivated by the fact that the search space plays a critical role in the NAS, we propose a novel and effective graph neural architecture search method called PSP from the perspective of search space design in this paper. We first propose an expressive search space composed of multiple cells. Instead of searching the entire architecture, we focus on searching the architecture of the cell. Then, we propose a progressive space pruning-based algorithm to search the architectures efficiently. Moreover, the data-specific search spaces and architectures obtained by PSP can be transferred to new graph datasets based on meta-learning. Extensive experimental results on different types of graph datasets reveal that PSP outperforms the state-of-the-art handcrafted architectures and the existing NAS methods in terms of effectiveness and efficiency.

![work_flow](https://github.com/PasaLab/PSP/blob/master/figures/work_flow.png)


### Requirements

Ensure that python version >= 3.6, pytorch version >= 1.4.0 . Then run:

```
pip install -r requirements.txt
```

+ **Notes:** This requirements file include torch_geometric == 1.6.3 (the key componet)

### Datasets

+ Dataset dir is fixed to `/data`
+ For Cora, Citeseer and Pubmed, we set the dataset dir as `data/planetoid/`, and the `torch_geometric.datasets.Planetoid` will download these three datasets automatically.
+ For chameleon, cornell, texas and wisconsin, these four datasets are from `new_data/` in [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn) . And the folder contains these datases is `data/heter_data/new_data/`
+ As is speaking to full_supervised split, [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn) provides 10 splits, as can be captured in `splits/`in [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). And to our folder, it is fixed in `data/heter_data/splits/`

> **Note:** For data from Geom-GCN, you can download from the link provide above.

### Search Architectures

To execute the PSP_NAS search process, please refer to psp_nas/README.md [here](https://github.com/PasaLab/PSP/tree/master/psp_nas).

### Evaluate Architectures

To evaluate the architectures search by PSP_NAS, please refer to evaluate/README.md [here](https://github.com/PasaLab/PSP/tree/master/evaluate).



### Case Study

In the paper, we demonstrate that we used the meta-learning method to match the proper models searched by psp_nas to online test i.e. quickly access to achieve high performance.

**Success Story:** This fast training framwork based on meta-learning help us rank the **2nd place** among hundreds of participants of the [KDD Cup AutoGraph challenge](https://www.4paradigm.com/content/details_85_1871.html).

