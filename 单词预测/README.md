# NLP单词预测

[TOC]

## 爬取数据

### 实验内容

从国内主流新闻网站（如腾讯、新浪、网易等）的科技频道抓取新闻内容。从国内主流新闻网站（如腾讯、新浪、网易等）的科技频道抓取新闻内容。要求新闻语言为中文，发布日期为2019年1月1日以后。数量至少为1000条。

### 实验过程

这是一个常见的数据采集过程，基本过程如下：

- 获取特定网页的url
- 下载网页的html
- 解析html中的中文信息
- 保存为文件

其中，获取网页的url的过程，可以采用链式爬取的规则，即后面生成的网页的$url_{i+1} $可以是当前爬取的网页出现的有效url。

提取中文信息，采用了BeautifulSoup库来解析html文本，由于本人爬取的是新浪的科技新闻网页，发现它的中文正文信息全在`<a>`下，所以很容易就提出了中文信息。

### 代码

这部分实验代码如下

```python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as np
from bs4 import BeautifulSoup
import requests
import re
def getText(url):
    
    ansStr = ''
    try:
        r = requests.get(url)
        r.encoding = r.apparent_encoding
        text = r.text
        
        soup = BeautifulSoup(text,'html.parser')
        
        for tag in soup.find_all('p'):
            try:
                for tagStr in tag.strings:
                   ansStr += tagStr
            except:
                print("error")
                continue
    except:
        print("can't access this ",url)

    return ansStr


def getUrls(url):
    
    try:
        r = requests.get(url)
        r.encoding = r.apparent_encoding
        text = r.text
        pageLinks = re.findall(r'(?<=<a href=\").*?(?=\")|(?<=href=\').*?(?=\')', text)
    except:
        print("can't access this url: ",url)
    
    ansList = []
    for x in pageLinks:
        if('tech.sina.com.cn' in x and '.shtml' in x ):
            y = x
            if('https' in x):
                y = x.replace('https','http')
            ansList.append(y)
    
    return ansList

def workAll(urlList,root,index = 1):
    for x in urlList:
        with open(root+str(index)+'.txt','w',encoding='utf-8') as f:
            f.write(x+'\n')
            f.write('\n')
            s = getText(x)
            f.writelines(s) 
        index += 1
    return index
def main():
    
    
    urlList = [ r'http://tech.sina.com.cn/zt_d/wic2019/',r'http://tech.sina.com.cn/news/',r'http://tech.sina.com.cn/it/',r'http://tech.sina.com.cn/tele/',\
        r'http://tech.sina.com.cn/elec/', r'http://tech.sina.com.cn/discovery/',r'http://tech.sina.com.cn/chuangshiji/',r'http://5g.sina.com.cn/']
    cnt = 1000
    for url0 in urlList:
        ans = getUrls(url0)
        print(ans)
        root = './NLP_news/'
        cnt = workAll(ans,root,cnt)
    
if __name__ == '__main__':
    main()
```

## 预处理数据

### 实验内容

对新闻数据进行预处理（分句、分词、去噪）。同时，在这个过程中同时要完成建立词表的工作

### 实验过程

该过程，将使用jieba包来进行分词处理。词与词之间用空格来分割，并且把每个句子单独占一行。

在分词同时，也同时建立词典。

### 实验代码

```python
import jieba

file_root = r'F:\NLP\expriment\exp2\NLP_news'
des_root = r'F:\NLP\expriment\exp3\data'

def checkValid(word):
    if(len(word) == 0):
        return False
    flag = False
    for x in word:
        #保留中文
        if(not('\u0030'<= x <= '\u0039')):
            flag = True
        #保留英文以及英文+数字的组合
        if(not(u'\u4e00'<= x <= u'\u9fff' or u'\u0061' <= x <= u'\u007a' \
           or u'\u0041'<=x<=u'\u005a' or '\u0030'<= x <= '\u0039')):
            return False
    if(not flag):
        return False
    return True

ans_dic = {}

for number in range(1,1249):
    file_name = file_root+'/'+str(number)+'.txt'
    des_file = des_root+'/'+str(number)+'.txt'
    try:
        with open(file_name,'r',encoding='utf-8')as f:
            s = f.readlines()
        s1 = s[2].replace('\u3000',' ')
        segList = jieba.cut(s1)
        s2 = ' '.join(segList)
        s3 = s2.replace('。','。\n')
        word_list = s3.split(' ')
        with open(des_file,'w',encoding='utf-8')as f:
            f.write(s3)
        for x in word_list:
            if(checkValid(x)):
                if(x not in ans_dic):
                    ans_dic[x] = 1
                else:
                    ans_dic[x] += 1
    except:
        continue

dic_file = des_root+'/词典5.txt'
index = 1
threhold = 2
with open(dic_file,'w',encoding='utf-8') as f:
    for x in ans_dic:
        if(ans_dic[x]>= threhold):
            f.write(str(index)+' '+ x +' '+str(ans_dic[x])+'\n')
            index += 1 
```

## 训练和测试语言模型

### 实验内容

使用预处理后的新闻数据训练两个语言模型。

- n-gram语言模型
- 基于LSTM的语言模型

并且使用提供的测试集检验训练好的模型的性能。具体来说，是要预测文本中的一个省缺词。

在提供的测试数据是在同一来源的新闻中随机抽取的100个句子，其中每个句子有一个词被挖空，要求预测出这些词。

例如：因为刷脸支付也得去打开手机接收验证码，所以还不如直接扫[MASK]更直接更方便。（ 标准答案：二维码）

### n-gram语言模型

#### 原理

假设我们有一个由n个词组成的句子$S=(w1,w2,⋯,wn)S=(w1,w2,⋯,wn)      S=(w_1,w_2,\cdots,w_n)S=(w1,w2,⋯,wn)$，如何衡量它的概率呢？让我们假设，每一个单词$w_i$都要依赖于从第一个单词$w_{1}$到它之前一个单词$w_{i-1}$的影响：
$$
P(S) = p(w_1w_2...w_n) = p(w_1)p(w_2|w_1)...p(w_n|w_{n-1}...w_2w_1)
$$
但是这样做，有很大的缺陷，体现在参数空间实在太大，为了实现简单，直接使用2-gram。即：
$$
P(S) = p(w_1w_2...w_n) = p(w_1)p(w_2|w_1)...p(w_n|w_{n-1})
$$


在具体实验中，条件概率又可以简单计算为：$P(w_i|w_{i-1}) = C(w_{i-1}w_i)/C(w_{i-1})$;其中$C(w_{i-1}w_i)$表示$w_{i-1}w_i$同时出现的次数。

在实验当中，就是要对一个句子中省缺的位置$k$，寻找到某个词，使得整个句子的概率达到最大。

即：$w = argmax_{w_k} P(S) =  p(w_1)p(w_2|w_1)...p(w_k|w_{k-1})p(w_{k+1}|w_k)...p(w_n|w_{n-1})$

其实等价为
$$
w = argmax_{w_k} p(w_k|w_{k-1})p(w_{k+1}|w_k) = C(w_{k-1}w_k)/C(w_{k-1}) * C(w_k w_{k+1})/C(w_k) 

\\ => C(w_{k-1}w_k) * C(w_k w_{k+1})/C(w_k)
$$


所以实验中，只需要分别统计上式中的三项。

### 实验过程

根据上述原理，对于每一次的需要预测的句子，提取出mask前一个词w1和后一个词w2,然后遍历所有的句子文本，统计每个这样的词w，它在w1之后，它出现在w2之前，同时统计C(w1w),C(ww2),然后根据上述公式，输出最大的10个预测值。

### 实验结果

![image-20191130214933019](NLP%E4%B8%AD%E6%9C%9F%E5%A4%A7%E4%BD%9C%E4%B8%9A.assets/image-20191130214933019.png)

### LSTM模型

### 原理

简单来说，它是一个基于RNN改进的模型，主要解决RNN的遗忘特性和梯度爆炸的问题。在这个问题中，我们是要预测一个句子中省缺的词，可以这样做，利用每一个词去预测下一个词，所以整个网络的架构就如下了（这也是课上所讲的）：

![image-20191130215231163](NLP%E4%B8%AD%E6%9C%9F%E5%A4%A7%E4%BD%9C%E4%B8%9A.assets/image-20191130215231163.png)

每个隐藏层的输出的最终结果就是需要预测的下一个词的结果。比如一个句子' \<BE> 我 喜欢 烧饼 \<END>'输入词为：['\<BE>','我', '喜欢', '烧饼'],则标签为['我', '喜欢', '烧饼','\<END>']。

### 实验过程

- 准备好训练数据和标签数据，编码句子。（Encode each word with a unique number）
- 搭建LSTM模型
- 调参训练
- 实验结果

首先，根据之前的实验得到的词表，把每个文本中的每个句子用唯一的整数来表示。得到word_to_ix和tag_to_ix，在这里word_to_ix和tag_to_ix是一样的（因为用词预测词，输入输出本来就是一样的词汇表）。

当然，在这个过程中，考虑到模型的准确性等因素，先去除了停用词。

之后，就是要来实现LSTM模型了。选择的框架是PyTorch，因为这个框架实际上在它的[官网](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html?highlight=lstm)都是有给的，所以实现起来，并不是很困难。

最后，选择在GPU上进行训练，时间不是很长。在训练集上的loss可以到0.07。

然而，在所提供的测试集上测试时，却发现，正确率为0，没有一个是预测正确的，这实在是非常令人失望。

```
科技 座椅
中 汽车
传输速度 欧洲
年 玻璃
政府 利润
赫尔曼 故宫
<END> 颁发
cyberbunker 广阔
量子 对话
美国 手机
方法 神经网络
印度人 机器人
中国 应用
囊中 增长
变速箱 学习
往往 门店
<END> 干燥
非法 政府
中有 厂商
可能 裁员
说 需求
<END> 他们
中国 鲜艳
左右开弓 打击
计算机 传销
气压 招聘
<END> 中心
年 人类
振动 程序
城市 班主任
暗网 浪费
月 建设
暗网 发展
股票 信息技术
道理 阶段
骗 公安局
信息 诽谤
地堡 商务部
年 关注
波音公司 服务
人才 成本
提供 泄露
年 运营商
世界 拖地
<END> 观众
相对论 基础
地堡 成员
大幅 改善
信息 资本
吸引 白酒
<END> 安全
赛博 零售业
斯文 宣布
印度 价格
赫尔曼 用户
直接 电池
存在 屏幕
<END> 汽车
年 企业
接触 时间
里 价格
地堡 面积
新 占比
运作 政策
管道 法律
技术 销量
更是 法律
产品 远
<END> 原因
蒸汽 发布
没 版本
万 因素
使用 影响
工作 媒体
产生 功能
没有 消费者
<END> 效率
首席 降
一家 产品
中国 人工智能
<END> 需求
现代 套餐
共 营收
可能 环境
低 资金
地堡 安全
级别 服务
科技 亏损
巨头 国家
做 损失
时 合作
美国 速度
<END> 人工智能
庇护 智能机
一个 产品
美国 无线电
<END> 营收
每个 安全
<END> 价格
是不是 企业
0
```

## 参考文献

[1] N-gram模型原理：https://blog.csdn.net/songbinxu/article/details/80209197 

[2] LSTM模型原理：https://blog.csdn.net/v_july_v/article/details/89894058

[3] LSTM模型的pytorch实现：https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html?highlight=lstm