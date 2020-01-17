#coding:utf-8
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#import torch.utils.data as Data
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Encoder ##########
# RNN编码一个句子序列
# 采用双向GRU
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, max_length, n_layers=1):
        super(EncoderRNN,self).__init__()
        
        # 定义网络框架
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers #网络层数
        self.max_length = max_length
        # embedding层
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        
        # n_layers层的lstm
        self.gru = nn.GRU(embedding_dim,hidden_dim,n_layers,bidirectional=True,batch_first=True)
        self.isBidirect = 2
        
        
    def forward(self,x_in,batch_size = 1,hidden=None):
        #x_in = x_in.view(batch_size,self.max_length,-1)
        output = self.embedding(x_in)
        #print(output.size())
        if(hidden is None):
            hidden = self.initHidden(batch_size)
        #print(hidden.size())
        output,hidden = self.gru(output,hidden)
        #print(hidden.size())
        hidden = torch.add(hidden[-1],hidden[-2])
        return output,hidden
    
    def initHidden(self,batch_size=1):
        return torch.zeros(self.isBidirect*self.n_layers,batch_size,self.hidden_dim,device=device)
        #return torch.zeros(self.isBidirect*self.n_layers,batch_size,self.hidden_dim,device=device)
    


##### RNN解码 ########

# RNN解码单个词语
class DecoderRNN(nn.Module):
    def __init__(self,hidden_dim,output_dim,n_layers=1):
        super(DecoderRNN,self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(output_dim,hidden_dim,)
        self.n_layers = n_layers
        
        self.gru = nn.GRU(hidden_dim,hidden_dim,n_layers,batch_first=True)
        self.out = nn.Linear(hidden_dim,output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,input_x,hidden=None,batch_size=1):
        
        output = self.embedding(input_x)
        output = output.view(1,-1,self.hidden_dim)
        
        output = F.relu(output)
        if(hidden is None):
            hidden = self.initHidden(batch_size)
        output,hidden = self.gru(output,hidden)
        
        output = self.softmax(self.out(output[0]))
        
        return output,hidden
    
    def initHidden(self,batch_size=1):
        #return torch.zeros(self.n_layers,batch_size,self.hidden_dim)
        return torch.zeros(self.n_layers,batch_size,self.hidden_dim,device=device)


# RNN Attention decoder #########

class AttnDecoderRNN(nn.Module):
    def __init__(self,hidden_dim,output_dim,max_length,n_layers=1):
        super(AttnDecoderRNN,self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim,hidden_dim,n_layers,batch_first=True)
        self.score = nn.Linear(2*hidden_dim,hidden_dim,bias=False)
        self.W_c = nn.Linear(3*hidden_dim,hidden_dim)
        self.out = nn.Linear(hidden_dim,output_dim)
        #self.softmax = nn.LogSoftmax(dim=2)
        self.max_length = max_length
        self.n_layers = n_layers
        
    def forward(self,input_x,encoder_outputs,hidden=None,batch_size=1):
        output = self.embedding(input_x)
        
        ## batch_size * 1 * embedding_dim
        output = output.view(batch_size,1,-1)
        output = F.relu(output)
        
        #hidden: n_layer* batch_size* hidden_dim
        if(hidden is None):
            hidden = self.initHidden(batch_size)
        hidden = hidden.view(self.n_layers,batch_size,self.hidden_dim)
        
        output,hidden = self.gru(output,hidden)
        ## hidden.size() = (n_layer,batch_size,hidden_dim)
              
        # encoder_outputs:batch_size * max_length * 2hidden_dim
        # score_out: batch_size * max_length * hidden
        score_out = self.score(encoder_outputs)
        score_out = torch.transpose(score_out,1,2)
        ## calculate the context vector
        score_out = torch.bmm(output,score_out)
        ## score_out: bacth_size * 1 * max_length
        alpha = F.softmax(score_out,dim=2)
        ## encoder_outputs: batch_size * max_length * 2hidden
        
        #c_t: batch_size * 1 * 2hidden_dim
        c_t = torch.bmm(alpha,encoder_outputs)

        output = torch.cat((c_t, output), dim=2)
        
        attn_h = torch.tanh(self.W_c(output))
        output = F.log_softmax(self.out(attn_h),dim=2)
        output = torch.transpose(output,1,2)
        
        return output,hidden
    def initHidden(self,batch_size=1):
        return torch.zeros(self.n_layers,batch_size,self.hidden_dim,device=device)