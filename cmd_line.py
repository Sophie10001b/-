import os
import argparse
from torch.backends import cudnn

from dataset_handel import dataset_load

def main(config):
    cudnn.benchmark=True

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)
    if not os.path.exists(config.dataset_path):
        os.makedirs(config.dataset_path)
    if not os.path.exists(config.dataset_graph_path):
        os.makedirs(config.dataset_graph_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    dataset_load(config)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Control the dataset import and module running.')

    parse.add_argument('--data_path',type=str,default='E:\研一\人工智能\实验\data',help='path to save dataset and result.')
    parse.add_argument('--dataset_path',type=str,default='E:\研一\人工智能\实验\data\dataset',help='path to save raw dataset.')
    parse.add_argument('--dataset_graph_path',type=str,default='E:\研一\人工智能\实验\data\dataset2graph',help='path to save dataset graph.')
    parse.add_argument('--result_path',type=str,default=r'E:\研一\人工智能\实验\data\result',help='path to save result.')

    parse.add_argument('--lr',type=float,default=1e-2,help='the learning rate for each module\'s optimizer.')
    parse.add_argument('--epochs',type=int,default=100,help='the total epochs for each module.')
    parse.add_argument('--module',type=str,default='GCN',choices=['GCN','GraphSAGE','GAT','GATs','all'],help='select module for training and testing.')
    parse.add_argument('--mode',default='training',choices=['download','process','training'],help='download for datasets; process for data2graph; training for module running.')
    parse.add_argument('--dataset',default='Cora',choices=['Cora','Citeseer','PubMed','Enzyme','PPI','LastFMAsia','Facebook','All_dataset'],help='choose a dataset for module running.')
    #parse.add_argument('--shuffle',type=bool,default=True,help='choose to use shuffle for each node training.')
    parse.add_argument('--test_gap',type=int,default=1,help='the gap between each test.')
    parse.add_argument('--PPI_label_type',type=int,default=0,choices=range(121),help='PPI label type for classify.')
    #parse.add_argument('--Enzyme_batch_size',type=int,default=90,help='Enzyme training batch size.')
    parse.add_argument('--head_num',type=int,default=1,help='GAT multi-head number, head <= 4.')

    config=parse.parse_args()
    main(config)