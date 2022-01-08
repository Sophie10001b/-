import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from Network_weight_init import weight_init

##forward负责提取每个节点的邻居信息，对应的图处理函数负责实现对应的单节点操作
class Attention(nn.Module):
    def __init__(self,input_channel,output_channel,dev,head=1):
        super(Attention,self).__init__()
        self.att_line=[]
        self.line=[]
        self.head=head
        self.att_line=nn.Linear(2*output_channel,1,device=dev,bias=False)
        self.line=nn.Linear(input_channel,output_channel,device=dev)
        #多头注意，此处仅采用最直接的重复线性层构造方法
        if head > 1:
            self.att_line1=nn.Linear(2*output_channel,1,device=dev,bias=False)
            self.line1=nn.Linear(input_channel,output_channel,device=dev)
        if head > 2:
            self.att_line2=nn.Linear(2*output_channel,1,device=dev,bias=False)
            self.line2=nn.Linear(input_channel,output_channel,device=dev)
        if head > 3:
            self.att_line3=nn.Linear(2*output_channel,1,device=dev,bias=False)
            self.line3=nn.Linear(input_channel,output_channel,device=dev)
        self.device=dev
    def forward(self,X,A):
        node_num,_=A.size()
        first=True
        load_list=range(node_num)
        for i in load_list:
            neibor_feature=X[A[i].bool()]
            if(first): result=Attention.att(self,X[i],neibor_feature); first=False
            else: result=torch.cat((result,Attention.att(self,X[i],neibor_feature)),dim=0)
        return result
    def att(self,xi,xj):
        node_num,feature_num=xj.size()
        #result=torch.tensor(0,dtype=torch.float,device=self.device,requires_grad=True)
        xj=torch.cat((xi.view(-1,feature_num),xj),dim=0)
        xi_bar=self.line(xi)
        xj_bar=self.line(xj)
        xi_bar=xi_bar.expand(xj_bar.size())
        aj=torch.cat((xi_bar,xj_bar),dim=1)
        aj=F.leaky_relu(self.att_line(aj)).view(-1)
        aj=F.softmax(aj,dim=0)
        xj_bar=torch.matmul(aj,xj_bar).view(1,-1)

        if self.head > 1:
            xi_bar1=self.line1(xi)
            xj_bar1=self.line1(xj)
            xi_bar1=xi_bar1.expand(xj_bar1.size())
            aj1=torch.cat((xi_bar1,xj_bar1),dim=1)
            aj1=F.leaky_relu(self.att_line1(aj1)).view(-1)
            aj1=F.softmax(aj1,dim=0)
            xj_bar1=torch.matmul(aj1,xj_bar1).view(1,-1)
            xj_bar=xj_bar+xj_bar1
        if self.head >2:
            xi_bar2=self.line2(xi)
            xj_bar2=self.line2(xj) 
            xi_bar2=xi_bar2.expand(xj_bar2.size())
            aj2=torch.cat((xi_bar2,xj_bar2),dim=1)
            aj2=F.leaky_relu(self.att_line2(aj2)).view(-1)
            aj2=F.softmax(aj2,dim=0)
            xj_bar2=torch.matmul(aj2,xj_bar2).view(1,-1)
            xj_bar=xj_bar+xj_bar2
        if self.head >3:
            xi_bar3=self.line3(xi)
            xj_bar3=self.line3(xj) 
            xi_bar3=xi_bar3.expand(xj_bar3.size())
            aj3=torch.cat((xi_bar3,xj_bar3),dim=1)
            aj3=F.leaky_relu(self.att_line3(aj3)).view(-1)
            aj3=F.softmax(aj3,dim=0)
            xj_bar3=torch.matmul(aj3,xj_bar3).view(1,-1)
            xj_bar=xj_bar+xj_bar3

        xj_bar=xj_bar/self.head
        #result=result+xj_bar
        #result=result/self.head_num
        return xj_bar


class GAT(nn.Module):
    def __init__(self,feature_num,output_channel1,output_channel2,class_num,dev,dataset_type,head=1,shuffle=True):
        super(GAT,self).__init__()
        self.device=dev
        self.shuffle=shuffle
        head1=1
        head2=1
        if head >1:
            head1=head
            head2=2
        self.Att1=Attention(feature_num,output_channel1,dev,head1)
        self.Att2=Attention(output_channel1,output_channel2,dev,head2)
        self.line1=nn.Linear(output_channel2,class_num,device=dev)
        self.Att1.apply(weight_init)
        self.Att2.apply(weight_init)
        self.line1.apply(weight_init)
        self.dataset_type=dataset_type
        self.class_num=class_num
        self.feature_num=feature_num
        self.output_channel1=output_channel1

    def forward(self,X,A):
        X=X.to(self.device)
        A=A.to(self.device)
        X=self.Att1(X,A)
        X=F.dropout(X,training=self.training)
        X=self.Att2(X,A)
        X=self.line1(X)
        if(self.dataset_type == 1):
            X=torch.mean(X,dim=0).view(1,-1)
        return F.log_softmax(X,dim=1)