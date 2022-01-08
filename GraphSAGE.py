import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from Network_weight_init import weight_init

#forward负责提取每个节点的邻居信息，对应的图处理函数负责实现对应的单节点操作
class Aggregate(nn.Module):
    def __init__(self,input_channel,output_channel,dev):
        super(Aggregate,self).__init__()
        self.neibor_line=nn.Linear(input_channel,output_channel,device=dev)
        self.line=nn.Linear(input_channel,output_channel,device=dev)
    def forward(self,X,A,neibor_num):
        first=True
        node_num,_=A.size()
        load_list=range(node_num)
        for i in load_list:
            neibor_feature=X[A[i].bool()]
            neibor_count,_=neibor_feature.size()
            #如果邻接节点数过多，直接按顺序将后续节点截断
            if(neibor_count>neibor_num):
                neibor_feature=neibor_feature[:neibor_num]
            if(first): result=Aggregate.aggr(self,X[i],neibor_feature); first=False
            else: result=torch.cat((result,Aggregate.aggr(self,X[i],neibor_feature)),dim=0)
        return result
    def aggr(self,xi,xj):
        node_num,feature_num=xj.size()
        if node_num > 0:
            xj=torch.mean(xj,dim=0)
            xj=F.leaky_relu(self.neibor_line(xj))
        xi=F.leaky_relu(self.line(xi))
        if node_num > 0:
            return (xi+xj).view(1,-1)
        else :
            return xi.view(1,-1)

class GraphSAGE(nn.Module):
    def __init__(self,neibor_num1,neibor_num2,feature_num,output_channel1,output_channel2,class_num,dev,dataset_type,shuffle=True):
        super(GraphSAGE,self).__init__()
        self.device=dev
        self.shuffle=shuffle
        self.Aggr1=Aggregate(feature_num,output_channel1,dev)
        self.Aggr2=Aggregate(output_channel1,output_channel2,dev)
        self.line1=nn.Linear(output_channel2,class_num,device=dev)
        self.Aggr1.apply(weight_init)
        self.Aggr2.apply(weight_init)
        self.line1.apply(weight_init)
        self.neibor_num1=neibor_num1
        self.neibor_num2=neibor_num2
        self.feature_num=feature_num
        self.class_num=class_num
        self.dataset_type=dataset_type

    def forward(self,X,A):
        X=X.to(self.device)
        A=A.to(self.device)
        X=self.Aggr1(X,A,self.neibor_num1)
        X=F.dropout(X,training=self.training)
        X=self.Aggr2(X,A,self.neibor_num2)
        X=self.line1(X)
        if(self.dataset_type == 1):
            X=torch.mean(X,dim=0).view(1,-1)
        return F.log_softmax(X,dim=1)
            



