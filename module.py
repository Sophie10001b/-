import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import numpy as np
import matplotlib.pyplot as plt
import joblib
import networkx as nx
from torch.serialization import load
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from tqdm.std import trange
from tqdm.contrib import tzip

from GCN import GCN
from GraphSAGE import GraphSAGE
from GAT import GAT

device=('cuda:0' if torch.cuda.is_available else 'cpu')


class module_training():
    def __init__(self,config,training_dataset):
        self.dataset=training_dataset
        self.path=config.result_path
        self.lr=config.lr
        self.epoch=config.epochs
        self.dataset_type=0
        self.eval_gap=config.test_gap
        #对不同数据集采用不同的维数设置
        if config.dataset == 'Cora': c1=700; c2=21; n1=10; n2=15
        elif config.dataset == 'Citeseer': c1=1851; c2=60; n1=10; n2=15
        elif config.dataset == 'PubMed': c1=240; c2=30; n1=10; n2=15
        elif config.dataset == 'Facebook': c1=64; c2=16; n1=10; n2=15
        elif config.dataset == 'Enzyme': c1=60; c2=30; n1=10; n2=15; self.dataset_type=1
        elif config.dataset == 'PPI': c1=25; c2=6; n1=10; n2=15; training_dataset.class_num=2
        elif config.dataset == 'LastFMAsia': c1=600; c2=363; n1=10; n2=15
        if(config.module == 'GCN'): self.module=GCN(training_dataset.feature_num,c1,c2,training_dataset.class_num,device,self.dataset_type).to(device)
        elif(config.module == 'GraphSAGE'): self.module=GraphSAGE(n1,n2,training_dataset.feature_num,c1,c2,training_dataset.class_num,device,self.dataset_type).to(device)
        elif(config.module == 'GAT'): self.module=GAT(training_dataset.feature_num,c1,c2,training_dataset.class_num,device,self.dataset_type).to(device)
        elif(config.module == 'GATs'): self.module=GAT(training_dataset.feature_num,c1,c2,training_dataset.class_num,device,self.dataset_type,head=config.head_num).to(device)
        self.optim=optm.Adam(self.module.parameters(),lr=config.lr)
        self.accuracy_list=[]
        self.loss_list=[]
        self.module_name=config.module
        self.dataset_name=config.dataset
        self.label_type=config.PPI_label_type

    def process(self):
        tepoch=trange(self.epoch)
        for epoch in tepoch:
            load_list=random.sample(range(self.dataset.graph_num),int(self.dataset.graph_num))
            loss=0
            total_loss=0
            loss_count=0
            self.module.train()
            self.optim.zero_grad()
            for i in load_list:
                graph=self.dataset.graph_struct[i]
                if(self.dataset.graph_num > 1 and graph['Test'] == 1): continue

                if(self.module_name == 'GCN'): 
                    Eye=torch.eye(graph['node_num'])
                    result=self.module(graph['D']+Eye,graph['X'],graph['A']+Eye)
                else: result=self.module(graph['X'],graph['A'])
                
                #对于不同结构的数据集采用不同的loss反向传播方式
                if(self.dataset.graph_num == 1):
                    loss=F.nll_loss(result[self.dataset.train_mask],graph['Y'][self.dataset.train_mask].long().to(device))
                    total_loss=loss.item()
                    loss_count+=1
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                elif(self.dataset_type == 1):
                    loss=F.nll_loss(result,graph['Y'].long().to(device))
                    total_loss+=loss.item()
                    loss_count+=1
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                else:
                    loss=F.nll_loss(result,graph['Y'][:,self.label_type].long().to(device))
                    total_loss+=loss.item()
                    loss_count+=1
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
            #if(self.dataset_type == 1):
            #    loss=F.nll_loss(result_list,self.dataset.raw_Y[self.dataset.train_mask].long().to(device))
            #    total_loss=loss
            #    loss.backward()
            #    self.optim.step()

            if((epoch+1) % self.eval_gap == 0):
                self.module.eval()
                with torch.no_grad():
                    if(self.dataset.graph_num == 1): accuracy=module_training.test(self,result[self.dataset.test_mask],graph['Y'][self.dataset.test_mask].to(device))
                    else: accuracy=module_training.test(self)
                    self.accuracy_list.append(accuracy)
                self.module.train()
            total_loss=total_loss/loss_count
            self.loss_list.append(total_loss)
            tepoch.set_description('module {0} is training in {1}, loss {2:.6}, accuracy {3:.6%}.'.format(self.module_name,self.dataset_name,total_loss,self.accuracy_list[-1]))
                    
    def test(self,result=None,label=None):
        accuracy=0
        if(self.dataset.graph_num == 1 and self.dataset_type == 0):
            test_result=torch.argmax(result,dim=1).view(-1)
            correct=float(test_result.eq(label).sum().item())
            accuracy=correct/list(label.size())[0]
            return accuracy
        #对于Enzyme数据集统计每张图的准确率
        elif(self.dataset.graph_num > 1 and self.dataset_type == 1):
            for graph in self.dataset.graph_struct:
                if(graph['Test'] == 1):
                    if self.module_name == 'GCN':
                        Eye=torch.eye(graph['node_num'])
                        result=self.module(graph['D']+Eye,graph['X'],graph['A']+Eye)
                    else:
                        result=self.module(graph['X'],graph['A'])
                    test_result=torch.argmax(result,dim=1).view(-1)
                    correct=float(test_result.eq(graph['Y'].to(device)).sum().item())
                    accuracy+=correct
            accuracy=accuracy/self.dataset.test_num
            return accuracy
        #对于PPI数据集统计测试图中所有节点的准确率
        elif(self.dataset.graph_num > 1 and self.dataset_type == 0):
            for graph in self.dataset.graph_struct:
                if(graph['Test'] == 1):
                    if self.module_name == 'GCN':
                        Eye=torch.eye(graph['node_num'])
                        result=self.module(graph['D']+Eye,graph['X'],graph['A']+Eye)
                    else:
                        result=self.module(graph['X'],graph['A'])
                    test_result=torch.argmax(result,dim=1).view(-1)
                    correct=float(test_result.eq(graph['Y'][:,self.label_type].to(device)).sum().item())
                    correct=correct/graph['node_num']
                    accuracy+=correct
            accuracy=accuracy/self.dataset.test_num
            return accuracy
        

def training(config,training_dataset):
    if config.module == 'all':
        if config.dataset in ['PubMed','Facebook','PPI','Enzyme']: module_list=['GCN','GraphSAGE','GAT']
        else: module_list=['GCN','GraphSAGE','GAT','GATs']
        module_loss_list=[]
        module_accuracy_list=[]
        result_file=os.path.join(config.result_path,config.module+'_'+config.dataset)
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        for i in module_list:
            start_time=time.perf_counter()
            config.module=i
            module=module_training(config,training_dataset)
            module.process()
            module_accuracy_list.append(module.accuracy_list)
            module_loss_list.append(module.loss_list)

            #存储每个模型的训练与测试信息
            with open(os.path.join(result_file,config.module+'_'+config.dataset+'_'+'acc_list'+'.txt'),'w') as f:
                loss_min_index=np.argmin(module.loss_list)
                loss_min=module.loss_list[loss_min_index]
                acc_max_index=np.argmax(module.accuracy_list)
                acc_max=module.accuracy_list[acc_max_index]
                info1='Module: {0}.  Dataset: {1}.  Best Loss: {2:.6} in {3} epoch.  Best Accuracy: {4:.6%} in {5} epoch.  training time: {6:.6}s'.format(config.module,config.dataset,loss_min,loss_min_index,acc_max,acc_max_index,time.perf_counter()-start_time)
                f.writelines(info1+'\n')

                total_param=sum(x.numel() for x in module.module.parameters())
                info2='Total parameters number: {}'.format(total_param)
                f.writelines(info2+'\n')
        
        #绘制每个模型的精度变化图
        print('all module training is finish, now start plotting...')
        plt.rcParams['figure.dpi']=600
        color_list=['r','g','b','y']
        color_list=color_list[:len(module_list)]
        for mod,acc,color in zip(module_list,module_accuracy_list,color_list):
            plt.plot(acc,label=mod,color=color,linewidth=0.5)
        plt.title('GNN accuracy in {}.'.format(config.dataset))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        if(os.path.exists(os.path.join(result_file,config.dataset+'_accuracy'+'.png'))):
                os.remove(os.path.join(result_file,config.dataset+'_accuracy'+'.png'))
        plt.savefig(os.path.join(result_file,config.dataset+'_accuracy'+'.png'),format='PNG')
        plt.close()

        for mod,los,color in zip(module_list,module_loss_list,color_list):
            plt.semilogy(los,label=mod,color=color,linewidth=0.5)
        plt.title('GNN loss in {}.'.format(config.dataset))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        if(os.path.exists(os.path.join(result_file,config.dataset+'_loss'+'.png'))):
                os.remove(os.path.join(result_file,config.dataset+'_loss'+'.png'))
        plt.savefig(os.path.join(result_file,config.dataset+'_loss'+'.png'),format='PNG')
        plt.close()

        config.module = 'all'

    else:
        result_file=os.path.join(config.result_path,config.module+'_'+config.dataset)
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        start_time=time.perf_counter()
        module=module_training(config,training_dataset)
        module.process()

        with open(os.path.join(result_file,config.module+'_'+config.dataset+'_'+'acc_list'+'.txt'),'w') as f:
            loss_min_index=np.argmin(module.loss_list)
            loss_min=module.loss_list[loss_min_index]
            acc_max_index=np.argmax(module.accuracy_list)
            acc_max=module.accuracy_list[acc_max_index]
            info1='Module: {0}.  Dataset: {1}.  Best Loss: {2:.6} in {3} epoch.  Best Accuracy: {4:.6%} in {5} epoch.  training time: {6:.6}s'.format(config.module,config.dataset,loss_min,loss_min_index,acc_max,acc_max_index,time.perf_counter()-start_time)
            f.writelines(info1+'\n')

            total_param=sum(x.numel() for x in module.module.parameters())
            info2='Total parameters number: {}'.format(total_param)
            f.writelines(info2+'\n')