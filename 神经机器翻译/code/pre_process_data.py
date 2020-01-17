#coding:utf-8
import numpy as np
import jieba
root_file = './data/'
train_file_name = root_file + r'train_source_8000.txt'
train_target_name = root_file + 'train_target_8000.txt'

with open(train_file_name,'r',encoding='utf-8')as f:
    strList = f.readlines()

## 分词
cut_source_file = root_file + 'cut_train_source_8000.txt'
with open(cut_source_file,'w',encoding = 'utf-8') as f:
    for line in strList:
        t = line.strip('\n')
        segList = jieba.cut(t)
        s = ' '.join(segList)
        f.write(s+'\n') 


file = root_file + 'test_source_1000.txt'
with open(file,'r',encoding='utf-8')as f:
    strList = f.readlines()
cut_test_file = root_file + r'cut_test_file.txt'
with open(cut_test_file,'w',encoding = 'utf-8') as f:
    for line in strList:
        t = line.strip('\n')
        segList = jieba.cut(t)
        s = ' '.join(segList)
        f.write(s+'\n') 

### 建立source的词典文件
def checkValid(word):
    if(len(word) == 0):
        return False
    flag = False
    for x in word:
        #保留英文以及英文+数字+中文
        if(not(u'\u4e00'<= x <= u'\u9fff' or u'\u0061' <= x <= u'\u007a' \
           or u'\u0041'<=x<=u'\u005a' or '\u0030'<= x <= '\u0039')):
            return False
    return True

source_dict_count = {}
source_dict_file = root_file + 'source_8000_dict.txt'

with open(cut_source_file,'r',encoding = 'utf-8') as f:
    strList = f.readlines()

with open(cut_test_file,'r',encoding = 'utf-8') as f:
    strList += f.readlines()    

for line in strList:
    t =  line.strip('\n').split(' ')
    for w in t:
        if(checkValid(w)):
            if w not in source_dict_count:
                source_dict_count[w] = 1
            else:
                source_dict_count[w] += 1

with open(source_dict_file,'w',encoding='utf-8')as f:
    cnt = 0
    for key,v in source_dict_count.items():
        f.write(str(cnt)+' '+str(key)+' '+str(v)+'\n')
        cnt += 1 

### 建立target的词典文件

target_dict_count = {}
target_dict_file = root_file + 'target_8000_dict.txt'
target_file = root_file + 'train_target_8000.txt'
target_test_file = root_file + 'test_target_1000.txt'
with open(target_file,'r',encoding='utf-8')as f:
    strList = f.readlines()
with open(target_test_file,'r',encoding='utf-8')as f:
    strList += f.readlines()
for line in strList:
    t = line.strip('\n').split(' ')
    for e in t:
        e = e.strip(',')
        e = e.strip('.')
        e = e.strip('!')
        e = e.strip('?')
        if(checkValid(e)):
            if e not in target_dict_count:
                target_dict_count[e] = 1
            else:
                target_dict_count[e] += 1
                
with open(target_dict_file,'w',encoding='utf-8')as f:
    cnt = 0
    for key,v in target_dict_count.items():
        f.write(str(cnt)+' '+str(key)+' '+str(v)+'\n')
        cnt += 1 