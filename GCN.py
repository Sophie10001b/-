import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from Network_weight_init import weight_init
'''
class Convolution(nn.Module):
    def __init__(self,input_channel,output_channel,dev):
        super(Convolution,self).__init__()
        self.line=nn.Linear(input_channel,output_channel,device=dev)
        self.device=dev
    def forward(self,D,H,A):
        D=torch.inverse(D)
        D=D.pow(0.5)
        D=(D @ A) @ D
        H=F.leaky_relu(self.line(D @ H))
        return H
'''
'''
class Convolution(nn.Module):
    def __init__(self,input_channel,output_channel,dev):
        super(Convolution,self).__init__()
        self.line=nn.Linear(input_channel,output_channel,device=dev)
        self.device=dev
    def forward(self,xj,dj):
        dj=torch.sqrt(dj[0]*dj)
        dj=(1/dj).view(1,-1)
        xj=dj @ xj
        xj=F.leaky_relu(self.line(xj))
        return xj.view(1,-1)
'''
#forward负责提取每个节点的邻居信息，对应的图处理函数负责实现对应的单节点操作
class Convolution(nn.Module):
    def __init__(self,input_channel,output_channel,dev):
        super(Convolution,self).__init__()
        self.line=nn.Linear(input_channel,output_channel,device=dev)
        self.device=dev
    def forward(self,D,X,A):
        first=True
        node_num,_=A.size()
        load_list=range(node_num)
        for i in load_list:
            neibor_feature=X[A[i].bool()]
            neibor_deg,_=torch.max(D[A[i].bool()],dim=1)
            if(first): result=Convolution.conv(self,neibor_feature,neibor_deg); first=False
            else: result=torch.cat((result,Convolution.conv(self,neibor_feature,neibor_deg)),dim=0)
        return result
    def conv(self,xj,dj):
        dj=torch.sqrt(dj[0]*dj)
        dj=(1/dj).view(1,-1)
        xj=dj @ xj
        xj=F.leaky_relu(self.line(xj))
        return xj.view(1,-1)

class GCN(nn.Module):
    def __init__(self,feature_num,output_channel1,output_channel2,class_num,dev,dataset_type,shuffle=True):
        super(GCN,self).__init__()
        self.Conv1=Convolution(feature_num,output_channel1,dev)
        self.Conv2=Convolution(output_channel1,output_channel2,dev)
        self.line1=nn.Linear(output_channel2,class_num,device=dev)
        self.Conv1.apply(weight_init)
        self.Conv2.apply(weight_init)
        self.line1.apply(weight_init)
        self.device=dev
        self.shuffle=shuffle
        self.dataset_type=dataset_type
        self.class_num=class_num
        self.feature_num=feature_num
        self.output_channel1=output_channel1

    def forward(self,D,X,A):
        D=D.to(self.device)
        X=X.to(self.device)
        A=A.to(self.device)
        X=self.Conv1(D,X,A)
        X=F.dropout(X,training=self.training)
        X=self.Conv2(D,X,A)
        X=self.line1(X)
        if(self.dataset_type == 1):
            X=torch.mean(X,dim=0).view(1,-1)
        return F.log_softmax(X,dim=1)
'''
    def forward(self,D,X,A):
        D=D.to(self.device)
        X=X.to(self.device)
        A=A.to(self.device)
        first=True
        node_num,_=A.size()
        if self.shuffle == True: load_list=random.sample(range(node_num),int(node_num))
        else: load_list=range(node_num)
        for i in load_list:
            neibor_feature=X[A[i].bool()]
            neibor_deg,_=torch.max(D[A[i].bool()],dim=1)
            if(first): Ha=self.Conv1(neibor_feature,neibor_deg); first=False
            else: Ha=torch.cat((Ha,self.Conv1(neibor_feature,neibor_deg)),dim=0)
        #Ha=Ha.view(node_num,-1)
        first=True
        for i in load_list:
            neibor_feature=Ha[A[i].bool()]
            neibor_deg,_=torch.max(D[A[i].bool()],dim=1)
            if(first): Hb=self.Conv2(neibor_feature,neibor_deg); first=False
            else: Hb=torch.cat((Hb,self.Conv2(neibor_feature,neibor_deg)),dim=0)
        #Hb=Hb.view(node_num,-1)
        Hb=self.line1(Hb)
        if(self.dataset_type == 1):
            X=torch.mean(X,dim=0).view(-1,self.class_num)
        return F.log_softmax(Hb,dim=1)
'''
    


