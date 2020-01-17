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

from module import EncoderRNN,DecoderRNN,AttnDecoderRNN
from prepare_data import prepare_sen
from train import train

def translate_predict(seq,encoder,decoder,target_to_ix,ix_to_target):
    encoder_output,encoder_hidden = encoder(torch.tensor(seq,dtype=torch.long,device=device).view(1,-1))
    
    ## decoder has 2 methods:
    ans = []
    # greedy
    decoder_input = torch.tensor([target_to_ix['SOS']],dtype=torch.long,device=device)
    decoder_hidden = encoder_hidden
    
    for di in range(100):
        decoder_output, decoder_hidden = decoder(
            decoder_input, encoder_output,decoder_hidden)
        #print(decoder_output,decoder_output.size())
        topv, topi = decoder_output.topk(1,dim=1)
        decoder_input = topi.squeeze()  # detach from history as input
        #print(decoder_input)
        if(decoder_input == target_to_ix['EOS']):
            break
        ans.append(decoder_input)
        decoder_input = torch.tensor([decoder_input],dtype=torch.long,device=device)
        
    # beam search
    en_language = [ ix_to_target[i] for i in ans]
    #print(en_language)
    torch.cuda.empty_cache()
    return ans,en_language

def main(file_version=3,isLoad=False,isTrain=False):
    ## 建立词典
    source_word_to_ix = {}
    ix_to_source_word = ['SOS','EOS','PAD']
    source_word_to_ix['SOS'] = 1
    source_word_to_ix['EOS'] = 2 
    source_word_to_ix['PAD'] = 0 
    target_to_ix = {}
    target_to_ix['PAD'] = 0
    target_to_ix['SOS'] = 1
    target_to_ix['EOS'] = 2
    ix_to_target = ['PAD','SOS','EOS']
    
    root_file = r'../data/'

    tempFile = root_file + 'source_8000_dict.txt'
    with open(tempFile,'r',encoding='utf-8')as f:
        strList = f.readlines()
    cnt = 3
    for line in strList:
        t = line.strip('\n').split(' ')
        source_word_to_ix[t[1]] = cnt
        ix_to_source_word.append(t[1])
        cnt += 1
    tempFile = root_file + 'target_8000_dict.txt'
    with open(tempFile,'r',encoding='utf-8')as f:
        strList = f.readlines()
    cnt = 3
    for line in strList:
        t = line.strip('\n').split(' ')
        target_to_ix[t[1]] = cnt
        ix_to_target.append(t[1])
        cnt += 1
    
    
    source_file = root_file + 'cut_train_source_8000.txt'
    target_file = root_file +'train_target_8000.txt'

    with open(source_file,'r',encoding='utf-8')as f:
        strList = f.readlines()

    with open(target_file,'r',encoding='utf-8')as f:
        strList2 = f.readlines()
    
    encoder_max_len = 40
    encoder_dic_len = len(ix_to_source_word)
    decoder_dic_len = len(ix_to_target)
    decoder_max_len = 40
    source_sen_list = prepare_sen(strList,source_word_to_ix,encoder_max_len)
    target_sen_list = prepare_sen(strList2,target_to_ix,decoder_max_len)
    
    ## 实例化网络模型
    
    epoch0 = 1000
    layers = 1
    hidden_size = 128
    batch_size = 70
    max_length = encoder_max_len
    lossList = []
    if(not isLoad):
        encoder = EncoderRNN(encoder_dic_len,32,hidden_size,max_length,layers)
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = AttnDecoderRNN(hidden_size,decoder_dic_len,max_length,layers)
        decoder = torch.nn.DataParallel(decoder).cuda()

    else:
        load_version = file_version
        encoder = torch.load(root_file + 'encoder'+str(load_version)+'.pkl')
        #encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.load(root_file + 'decoder'+str(load_version)+'.pkl')
    #return encoder,decoder,lossList
    ### 训练
    if(isTrain):
        lossList,encoder,decoder = train(encoder,decoder,source_sen_list,target_sen_list,source_word_to_ix,target_to_ix,\
                         decoder_max_len,batch_size=batch_size,epoch0=epoch0)
        file_version += 1
        torch.save(encoder, root_file + 'encoder'+ str(file_version) + '.pkl')  # save entire net
        torch.save(encoder.state_dict(), root_file + 'encoder_params'+ str(file_version) + '.pkl') 
        torch.save(decoder,root_file + 'decoder'+ str(file_version) + '.pkl')
        torch.save(decoder.state_dict(),root_file + 'decoder_params'+ str(file_version)+ '.pkl')
        ### 画图
        from matplotlib import pyplot as plt

        epochList = [i+1 for i in range(len(lossList))]

        plt.figure()
        plt.plot(epochList,lossList,'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()


    ### 预测结果
    
    test_source_file = root_file + r'cut_test_file.txt'
    test_target_file = root_file + r'test_target_1000.txt'
    
    with open(test_source_file,'r',encoding='utf-8')as f:
        strList = f.readlines()
        
    with open(test_target_file,'r',encoding='utf-8')as f:
        strList2 = f.readlines()
    
    source_test_list = prepare_sen(strList,source_word_to_ix,encoder_max_len)
    target_test_list = prepare_sen(strList2,target_to_ix,decoder_max_len)
    ave = 0.0
    
    ## 预测
    score_list = []
    pre_result = []
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    from nltk.translate.bleu_score import sentence_bleu
    for index,sen in enumerate(source_test_list):
        #print("sen:",sen)
        ans,eng = translate_predict(sen,encoder,decoder,target_to_ix,ix_to_target)
        ground_truth = []
        #print("index",index)
        for x in target_test_list[index]:
            if(ix_to_target[x] == 'PAD'):
                break
            ground_truth.append(ix_to_target[x])
        smooth = SmoothingFunction()
        
        tt = sentence_bleu([eng],ground_truth,smoothing_function=smooth.method3)
        pre_result.append([eng,ground_truth,tt])
        score_list.append(tt)


    ave = sum(score_list)/len(score_list)

    print("average bleu is ",ave)
  

    return encoder,decoder,lossList,score_list,pre_result

import sys
if __name__ == '__main__':
    root_file = r'../data/'
    isLoad = False
    isTrain = True
    if(len(sys.argv) == 3):
        if(int(sys.argv[1]) == 1):
            isLoad = True
        if(int(sys.argv[2]) == 0):
            isTrain = False
    print(isLoad,isTrain)
    encoder,decoder,lossList,score_list,pre_result = main(7,isLoad,isTrain)
    print("the score is",score_list)
    with open(root_file + 'pre_result.txt','w',encoding='utf-8')as f:
        for sen,ground_truth,tt in pre_result:
            f.write('predict:' + str(sen)+'\n ground_truth: '+str(ground_truth) + '\n bleu: ' + str(tt) + '\n')
    