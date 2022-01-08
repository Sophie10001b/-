import os
import gc
import time
import random
import torch
import numpy as np
import joblib
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import LastFMAsia
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import PPI

from module import training

#从PYG下载数据集
class datasets():
    def __init__(self):
        pass

    def download(self,dataset_path):
        cora_dataset=Planetoid(root=dataset_path,name='Cora')
        citeseer_dataset=Planetoid(root=dataset_path,name='CiteSeer')
        pubmed_dataset=Planetoid(root=dataset_path,name='PubMed')
        enzyme_dataset=TUDataset(root=dataset_path,name='ENZYMES')
        ppi_dataset=PPI(root=os.path.join(dataset_path,'PPI'))
        fmasia_dataset=LastFMAsia(root=os.path.join(dataset_path,'LastFMAsia'))
        facebook_dataset=FacebookPagePage(root=os.path.join(dataset_path,'Facebook'))
        self.dataset_list={'cora_dataset':cora_dataset,'citeseer_dataset':citeseer_dataset,'pubmed_dataset':pubmed_dataset,'enzyme_dataset':enzyme_dataset,'ppi_dataset':ppi_dataset,'fmasia_dataset':fmasia_dataset,'facebook_dataset':facebook_dataset}
        with open(os.path.join(dataset_path,'download_raw_data'),'wb') as f:
            joblib.dump(self.__dict__,f,compress=('gzip',3))
    
    def load_raw_data(self,dataset_path):
        with open(os.path.join(dataset_path,'download_raw_data'),'rb') as f:
            self.__dict__.update(joblib.load(f))


#引文网络与社交网络结构相同，使用同一个类进行处理
class SingleNetwork2graph():
    def __init__(self,network_name):
            self.data_name=network_name
    
    def handle(self,network,dataset_graph_path,test_percent=0.3):
        data=network
        self.class_num=data.num_classes
        self.graph_num=data.len()
        self.graph_struct=[]
        data=data[0]
        Xn=data.x
        Yn=data.y
        node_num,self.feature_num=data.x.size()

        test_index=torch.tensor(random.sample(range(node_num),int(node_num*test_percent)))
        train_mask=torch.ones(node_num,dtype=torch.bool)
        test_mask=torch.zeros(node_num,dtype=torch.bool)
        for i in test_index.numpy():
            train_mask[i]=False; test_mask[i]=True
        self.train_mask=train_mask
        self.test_mask=test_mask
        self.test_num=int(node_num*test_percent)

        forward=data.edge_index[0]
        backward=data.edge_index[1]
        adj_matrix=torch.zeros((node_num,node_num))
        for i,j in zip(forward,backward):
            adj_matrix[i][j]=adj_matrix[j][i]=1
        An=adj_matrix
        Dn=torch.diag(torch.sum(An,dim=1))
        graph_struct={'node_num':node_num, 'X':Xn, 'Y':Yn, 'A':An, 'D':Dn, 'Test':-1}
        self.graph_struct.append(graph_struct)

        with open(os.path.join(dataset_graph_path,self.data_name),'wb') as f:
            joblib.dump(self.__dict__,f,compress=('gzip',3))

    def load(self,dataset_graph_path):
         with open(os.path.join(dataset_graph_path,self.data_name),'rb') as f:
            self.__dict__.update(joblib.load(f))


#蛋白质互作网络与酶网络相同，使用同一个类进行处理
class MultiNetwork2graph():
    def __init__(self,network_name):
            self.data_name=network_name
    
    def handle(self,network,dataset_graph_path,test_percent=0.3):
        data=network
        self.class_num=data.num_classes
        self.feature_num=data.num_features
        self.graph_num=data.len()
        self.graph_struct=[]
        edge_slice=data.slices['edge_index']
        node_slice=data.slices['x']
        label_slice=data.slices['y']
        data=data.data
        self.raw_Y=data.y

        test_index=torch.tensor(random.sample(range(self.graph_num),int(self.graph_num*test_percent)))
        train_mask=torch.ones(self.graph_num,dtype=torch.bool)
        test_mask=torch.zeros(self.graph_num,dtype=torch.bool)
        for i in test_index.numpy():
            train_mask[i]=False; test_mask[i]=True
        self.train_mask=train_mask
        self.test_mask=test_mask
        self.test_num=int(self.graph_num*test_percent)

        for i in range(1,self.graph_num+1):
            node_num=node_slice[i]-node_slice[i-1]
            edge_index=data.edge_index[:,edge_slice[i-1]:edge_slice[i]]
            Xn=data.x[node_slice[i-1]:node_slice[i],:]
            Yn=data.y[label_slice[i-1]:label_slice[i]]
            forward=edge_index[0]
            backward=edge_index[1]
            adj_matrix=torch.zeros((node_num,node_num))
            for a,b in zip(forward,backward):
                adj_matrix[a][b]=adj_matrix[a][b]=1
            An=adj_matrix
            Dn=torch.diag(torch.sum(An,dim=1))
            #包含多张图，每一张图对应一个dict，分别存储图的节点数，邻接矩阵，特征矩阵，标签和度矩阵
            graph_struct={'node_num':node_num, 'X':Xn, 'Y':Yn, 'A':An, 'D':Dn, 'Test':int(self.test_mask[i-1])}
            self.graph_struct.append(graph_struct)

        with open(os.path.join(dataset_graph_path,self.data_name),'wb') as f:
            joblib.dump(self.__dict__,f,compress=('gzip',3))

    def load(self,dataset_graph_path):
         with open(os.path.join(dataset_graph_path,self.data_name),'rb') as f:
            self.__dict__.update(joblib.load(f))


def dataset_load(config):
    data=datasets()

    #下载数据集
    if config.mode == 'download':
        data.download(config.dataset_path)

    #依次处理数据集并保存
    elif config.mode == 'process':
        data.load_raw_data(config.dataset_path)

        start_time=time.perf_counter()
        print('Processing Cora...')
        Cora=SingleNetwork2graph('Cora')
        Cora.handle(data.dataset_list['cora_dataset'],config.dataset_graph_path)
        del Cora; gc.collect()
        print('Cora process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

        start_time=time.perf_counter()
        print('Processing Citeseer...')
        Citeseer=SingleNetwork2graph('Citeseer')
        Citeseer.handle(data.dataset_list['citeseer_dataset'],config.dataset_graph_path)
        del Citeseer; gc.collect()
        print('Citeseer process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

        start_time=time.perf_counter()
        print('Processing PubMed...')
        PubMed=SingleNetwork2graph('PubMed')
        PubMed.handle(data.dataset_list['pubmed_dataset'],config.dataset_graph_path)
        del PubMed; gc.collect()
        print('PubMed process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

        start_time=time.perf_counter()
        print('Processing Enzyme...')
        Enzyme=MultiNetwork2graph('Enzyme')
        Enzyme.handle(data.dataset_list['enzyme_dataset'],config.dataset_graph_path)
        del Enzyme; gc.collect()
        print('Enzyme process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

        start_time=time.perf_counter()
        print('Processing PPI...')
        PPi=MultiNetwork2graph('PPI')
        PPi.handle(data.dataset_list['ppi_dataset'],config.dataset_graph_path)
        del PPi; gc.collect()
        print('PPI process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)
        
        start_time=time.perf_counter()
        print('Processing LastFMAsia...')
        FMAsia=SingleNetwork2graph('LastFMAsia')
        FMAsia.handle(data.dataset_list['fmasia_dataset'],config.dataset_graph_path)
        del FMAsia; gc.collect()
        print('LastFMAsia process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

        start_time=time.perf_counter()
        print('Processing Facebook...')
        Facebook=SingleNetwork2graph('Facebook')
        Facebook.handle(data.dataset_list['facebook_dataset'],config.dataset_graph_path)
        del Facebook; gc.collect()
        print('Facebook process is finish, spend time {:.6}s.'.format(time.perf_counter()-start_time))
        time.sleep(2)

    #使用指定模型与数据集训练
    elif config.mode == 'training':
        all_dataset_label=False
        if config.dataset == 'All_dataset':
            dataset_list=['Cora','Citeseer','PubMed','LastFMAsia','Facebook','Enzyme','PPI']
            all_dataset_label=True
        else:
            dataset_list=[config.dataset]
        for i in dataset_list:
            config.dataset=i
            #if i in ['Cora','Citeseer'] and config.module == 'all' and all_dataset_label == True:
            #    config.epochs=100
            #elif i in ['LastFMAsia','PubMed','PPI'] and config.module == 'all' and all_dataset_label == True:
            #    config.epochs=75
            #    config.head_num=2
            #elif i in ['Facebook','Enzyme'] and config.module == 'all' and all_dataset_label == True:
            #    config.epochs=50
            #    config.head_num=2
            print('{} is loading...'.format(config.dataset))
            if config.dataset in ['Cora','Citeseer','PubMed','LastFMAsia','Facebook']: training_dataset=SingleNetwork2graph(config.dataset)
            elif config.dataset in ['Enzyme','PPI']: training_dataset=MultiNetwork2graph(config.dataset)
            training_dataset.load(config.dataset_graph_path)
            print('{} load finish !'.format(config.dataset))
            training(config,training_dataset)



if __name__ == '__main__':
    #config=argparse.Namespace(mode='process',dataset_path='E:\研一\人工智能\实验\data\dataset',dataset_graph_path='E:\研一\人工智能\实验\data\dataset2graph')
    '''
    config=argparse.Namespace(
        mode='process',
        dataset='Cora',
        dataset_path='E:\研一\人工智能\实验\data\dataset',
        dataset_graph_path='E:\研一\人工智能\实验\data\dataset2graph',
        result_path=r'E:\研一\人工智能\实验\data\result',
        lr=1e-2,
        epochs=10,
        module='all',
        shuffle=False,
        test_gap=1,
        PPI_label_type=0,
        Enzyme_batch_size=90)
    dataset_load(config)
    '''