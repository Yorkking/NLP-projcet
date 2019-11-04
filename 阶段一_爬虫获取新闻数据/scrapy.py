# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:00:02 2019

@author: York_king
"""

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
    
    '''
    url0 = r'http://tech.sina.com.cn/discovery/'
    
    text = getText(url0)
    file = 'test1.txt'
    print(text)
    with open(file,'w',encoding='utf-8') as f:
        f.write(text)
    '''
    #url0 = r'http://tech.sina.com.cn/zt_d/wic2019/'
    
    #url0 = r'http://tech.sina.com.cn/it/'
    
    #urlList = [r'http://tech.sina.com.cn/tele/',r'http://tech.sina.com.cn/elec/', \
    # r'http://tech.sina.com.cn/discovery/',r'http://tech.sina.com.cn/chuangshiji/',r'http://5g.sina.com.cn/']
    urlList = [r'http://tech.sina.com.cn/news/']
    cnt = 970
    for url0 in urlList:
        ans = getUrls(url0)
        print(ans)
        root = './NLP_news/'
        cnt = workAll(ans,root,cnt)
    

if __name__ == '__main__':
    main()







