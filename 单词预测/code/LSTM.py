#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
torch.manual_seed(1)

def getStopWord(file):
    with open(file,'r',encoding='utf-8')as f:
        words = f.readlines()
        temp = [w[:-1] for w in words]
        ans = {}
        for w in temp:
            ans[w] = 1
        return ans
def getWordIX(word_ix_file,stopWords):
    #word_ix_file = r'F:\NLP\expriment\exp3\data\词典5.txt'
    word_to_ix = {}
    ix_to_word = []
    cnt = 0
    with open(word_ix_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            temp= line.split()
            if(temp[1] not in stopWords):
                ix_to_word.append(temp[1])
                word_to_ix[temp[1]] = cnt
                cnt += 1
    
    return word_to_ix,ix_to_word
def showSeq(ix_seq,ix_to_word):
    ans = ''
    for x in ix_seq:
        ans += ix_to_word[x]+' '
    return ans

def predictIx(model,test,ix_to_word):
    with torch.no_grad():
        inputs = prepare_sequence(test)
        tag_scores = model(inputs)
        ans = ''
        for line in tag_scores:
            index = torch.argmax(line)
            ans += ix_to_word[index]+' '
        return ans

def prepareData(fileRoot=r'F:\NLP\expriment\exp3\data'，word_to_ix):
    #fileRoot = r'F:\NLP\expriment\exp3\data'
    trainData = []
    for i in range(1,1249):
        try:
            fileName = fileRoot + '/'+ str(i)+ '.txt'
            with open(fileName,'r',encoding='utf-8') as f:
                sentences = f.readlines()
                for line in sentences:
                    sen = [word_to_ix['<BE>']]
                    for w in line.split():
                        if(w in word_to_ix):
                            sen.append(word_to_ix[w])
                    sen.append(word_to_ix['<END>'])
                    trainData.append((sen[0:-1],sen[1:]))
        except:
            continue
                
    return trainData
    
def prepare_sequence(seq):
    #idxs = [to_ix[w] for w in seq]
    return torch.tensor(seq, dtype=torch.long)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
def main():   
    stopWords = getStopWord(r'F:\NLP\expriment\stopwords\中文停用词表.txt')
    word_to_ix,ix_to_word = getWordIX(r'F:\NLP\expriment\exp3\data\词典5.txt',stopWords)
    tag_to_ix = word_to_ix
    trainData = prepareData(word_to_ix)

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32
    print(len(word_to_ix),len(tag_to_ix))
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    model = torch.nn.DataParallel(model).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    cnt = 0
    batch_size = 500
    start_i = random.randint(0,len(trainData)//batch_size-1)
    traningData = trainData[start_i*batch_size:(start_i+1)*batch_size]
    for epoch in range(300):
        for sentence, tags in traningData:

            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence)
            sentence_in = sentence_in.cuda()
            targets = prepare_sequence(tags)
            targets = targets.cuda()
            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            if(cnt%1000 == 0):
                print("loss",loss)
            cnt += 1
            loss.backward()     
            optimizer.step()

    with open(r'./questions1.txt','r',encoding='utf-8')as f: 
        seqs = f.readlines()
    predict_ans = []
    L = len(word_to_ix)
    mask = random.randint(0,30171)
    for line in seqs:
        temp = line.split()
        seq = []
        for index,v in enumerate(temp):
            if(v == '[' or v == ']'):
                continue
            elif(v == '，'):
                seq.append(word_to_ix['<BE>'])
            elif(v == '。'):
                seq.append(word_to_ix['<END>'])
            elif(v == 'MASK'):
                index0 = len(seq)
                seq.append(mask)
            elif(v in stopWords):
                continue
            elif(v not in word_to_ix):
                seq.append(random.randint(0,L))
            else:
                seq.append(word_to_ix[v])
        b = predictIx(model,seq,ix_to_word)
        predict_ans.append(b.split()[index0])

    with open(r'./answer.txt','r',encoding='utf-8')as f:
        a = f.readlines()
    answer = [x[:-1] for x in a]
    cnt = 0
    for index in range(100):
        print(predict_ans[index],answer[index])
        if(predict_ans[index] == answer[index]):
            cnt += 1
    print("correct number is: ",cnt)   
if __name__ == '__main__':
    main()
    