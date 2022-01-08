# 人工智能课程设计：GCN,GraphSAGE,GAT的手动实现以及在7类数据集上的运行结果
本实验基于PyTorch框架简易实现GCN,GraphSAGE,GAT三类图网络的逐节点运算过程，基于简易实现考虑，未使用batch，GAT多头注意机制仅采用固定数量实现

代码运行于如下模块中：
> \- python 3.7.10  
\- pytorch 1.9.1    
\- pyg 2.0.2

## 数据集下载
通过torch_geometric直接下载以下数据集：
> \- Cora   
\- Citeseer  
\- PubMed  
\- LastFMAsia  
\- Facebook  
\- Enzyme  
\- PPI

其中Enzyme为多图节点分类数据集，PPI为多图图分类数据集。使用以下命令下载数据集：
```
python cmd_line.py --data_path *** --dataset_path *** --dataset_graph_path *** --result_path *** --mode download
```
其中 *data_path* 后为保存文件根目录， *dataset_path* 后为源数据集下载目录， *dataset_graph_path* 后为源数据预处理后的保存目录， *result_path* 后为模型运行结果保存目录，推荐运行前在 *cmd_line.py* 中直接更改对应的默认设置。后续命令将省略以上的路径设置 

## 数据集处理
下载完成数据集后，本程序统一将每一张图处理成 **节点数；节点特征；分类标签；邻接矩阵；度矩阵** 的字典格式，并保存至 *dataset_graph_path* 下，其运行命令如下：
```
python cmd_line.py --mode process
```

## 模型训练以及结果输出
本程序通过以下命令使用对应数据集与模型进行训练：
```
python cmd_line.py --mode training --module *** --dataset ***
```
其中 *module* 包含以下选项
> \- GCN  
\- GraphSAGE  
\- GAT  
\- GATs  
\- all

- *GATs* 即采用多头注意的GAT模型，其第一层固定使用4头注意，第二层固定使用2头注意  
- *all* 即按照以上顺序对指定数据集依次使用模型训练

*dataset* 包含以下选项：
> \- Cora  
\- Citeseer  
\- PubMed  
\- LastFMAsia  
\- Facebook  
\- Enzyme  
\- PPI  
\- All_dataset

- *All_dataset* 即按照以上顺序使用指定模型依次训练

最后，模型会输出运行时的loss折线图与精度折线图，同时得到一个包含有模型名称，数据集名称，最优loss，最优精度，训练耗时，总参数数量的文本文件作为运行结果