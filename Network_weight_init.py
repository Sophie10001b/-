import torch
import torch.nn as nn
import random

random.seed(1)
torch.manual_seed(1)

#对于每个模型的MLP层进行参数初始化，其中权重为(0,0.1)的正态分布，偏置为(1,0.1)的正态分布
def weight_init(m):
    classname=m.__class__.__name__
    if(classname.find('Convolution'))!=-1:
        m.line.apply(weight_init)
    elif(classname.find('Aggregate'))!=-1:
        m.neibor_line.apply(weight_init)
        m.line.apply(weight_init)
    elif(classname.find('Attention'))!=-1:
            m.att_line.apply(weight_init)
            m.line.apply(weight_init)
            if m.head >1:
                m.att_line1.apply(weight_init)
                m.line1.apply(weight_init)
            if m.head >2:
                m.att_line2.apply(weight_init)
                m.line2.apply(weight_init)
            if m.head >3:
                m.att_line3.apply(weight_init)
                m.line3.apply(weight_init)
    elif(classname.find('Linear'))!=-1:
        nn.init.normal_(m.weight.data,0.0,0.1)
        if m.bias == None: pass
        else: nn.init.normal_(m.bias.data,1.0,0.1) 