# coding:utf-8
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

def prepare_sen(sentenceList,dic,length=None,isTensor=False):
    ans = []
    if(length is None):
        for line in sentenceList:
            t = line.strip('\n').split(' ')
            ans_e = [dic['SOS']]
            cnt = 1
            for index,e in enumerate(t):
                e = e.strip(',')
                e = e.strip('.')
                e = e.strip('!')
                e = e.strip('?')
                if(e not in dic):
                    continue
                ans_e.append(dic[e])
                cnt += 1
            ans_e.append(dic['EOS'])
            #ans_e = torch.tensor(ans_e,dtype=torch.long,device=device).view(1,-1)
            ans.append(ans_e)
    else:
        for line in sentenceList:
            t = line.strip('\n').split(' ')
            ans_e = [dic['PAD'] for _ in range(length)]
            ans_e[0] = dic['SOS']
            cnt = 1
            for index,e in enumerate(t):
                e = e.strip(',')
                e = e.strip('.')
                e = e.strip('!')
                e = e.strip('?')
                
                if(e not in dic):
                    continue
                try:
                    ans_e[cnt] = dic[e]
                except:
                    ans_e[cnt-1] = dic['EOS']
                    break
                cnt += 1
            
            if isTensor:
                ans_e = torch.tensor(ans_e,dtype=torch.long,device=device).view(1,-1)
            ans.append(ans_e)
    return ans