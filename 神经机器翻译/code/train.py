
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
import numpy as np
def train(encoder,decoder,source_sen_list,target_sen_list,source_word_to_ix,target_to_ix,\
          decoder_max_len,batch_size=70,learning_rate=0.003,layers=1,epoch0=200):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_list = []
    #layers = 1
    #hidden_size = 32
    #max_length = encoder_max_len
    '''
    encoder = EncoderRNN(encoder_dic_len,32,hidden_size,max_length,layers)
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = AttnDecoderRNN(hidden_size,decoder_dic_len,max_length,layers)
    decoder = torch.nn.DataParallel(decoder).cuda()
    '''
    #batch_size = 70
    #learning_rate = 0.03
    '''
    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
    '''
    encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=0)
    teacher_forcing_ratio = 0.3
    index_list = np.array([i for i in range(len(target_sen_list))])
    
    for epoch in range(epoch0):

        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        batch_start_index = random.randint(0,len(source_sen_list)//batch_size)
        batch_start_index = 1
        np.random.shuffle(index_list)
        '''
        train_data = [source_sen_list[x] for x in index_list[0:batch_size]]
        train_label = [target_sen_list[x] for x in index_list[0:batch_size]]
        
        source_data = torch.tensor(train_data,device=device)
        target_data = torch.tensor(train_label,device=device)
        '''
        
        source_data = torch.tensor(source_sen_list[batch_start_index*batch_size:(batch_start_index+1)*batch_size],device=device)
        target_data = torch.tensor(target_sen_list[batch_start_index*batch_size:(batch_start_index+1)*batch_size],device=device)
        
        encoder_out,encoder_hidden = encoder(source_data,batch_size)
        decoder_hidden = encoder_hidden

        decoder_input = torch.tensor([[target_to_ix['SOS']] for _ in range(batch_size)],device=device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for index in range(decoder_max_len-1):

                decoder_out,decoder_hidden = decoder(\
                   decoder_input, encoder_out, decoder_hidden,batch_size)

                loss += criterion(decoder_out,target_data[:,index+1].view(-1,1))
                decoder_input = target_data[:,index+1]
        else:
            for index in range(decoder_max_len-1):
                
                decoder_out,decoder_hidden = decoder(\
                   decoder_input, encoder_out, decoder_hidden,batch_size)
                loss += criterion(decoder_out,target_data[:,index+1].view(-1,1))
                topv, topi = decoder_out.topk(1,dim=1)
                decoder_input = topi.squeeze().detach().view(-1,1) # detach from history as input
                
                

        loss_list.append(loss/batch_size)
        if(epoch % 50 == 0):
            print("epoch:",epoch,"loss:",loss/batch_size)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        torch.cuda.empty_cache()

    return loss_list,encoder,decoder

