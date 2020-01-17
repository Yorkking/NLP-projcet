#coding='utf-8'
import numpy as np
import jieba

def predictWord(predictStr,fileRoot):
    desStr = "/".join(jieba.cut(predictStr, cut_all=False))  # 精确模式
    wordList = desStr.split('/')
    for index in range(len(wordList)):
        if(wordList[index] == '['):
            w1 = wordList[index-1]
            w2 = wordList[index+3]
            break
    
    ## 打开文件的过程
    fileNum = 1249
    cw1A = {}
    cAw2 = {}
    cA =  {}
    for num in range(1,fileNum):
        try:
            with open(fileRoot+'/'+str(num)+'.txt','r',encoding = 'utf-8') as f:
                try:
                    tempWordList = f.read().split(' ')

                    for index in range(len(tempWordList)):
                        if(tempWordList[index] == w1):
                            if(index+1<len(tempWordList)):
                                A = tempWordList[index+1]
                                if(A in cw1A):
                                    cw1A[A] += 1
                                else:
                                    cw1A[A] = 1
                                if(A  in cA):
                                    cA[A] += 1
                                else:
                                    cA[A] = 1

                        if(tempWordList[index] == w2):
                            if(index-1>=0):
                                A = tempWordList[index-1]
                                if(A in cAw2):
                                    cAw2[A] += 1
                                else:
                                    cAw2[A] = 1
                                if(A in cA):
                                    cA[A] += 1
                                else:
                                    cA[A] = 1

                except:
                    continue
        except:
            continue
                        
    ### 计算概率，输出最大概率的10个
    resultList = []
    for word in cA:
        try:
            p = cw1A[word] * cAw2[word] / cA[word]
            resultList.append((p,word))
        except:
            resultList.append((0,word))
    resultList.sort(reverse=True)
    
    ans = []
    for y,x in resultList:
        if(len(ans) == 10):
            break
        if('，' in x or '\n' in x or '。' in x or x == '' or x in "“" or x in "”"):
            continue
        #print(x)
        ans.append(x)
    return ans
    
def main():
    fileRoot = r'F:\NLP\expriment\exp3\data'
    questionFile = r'F:\NLP\expriment\questions.txt'
    with open(questionFile,'r',encoding = 'utf-8')as f:
        qList = f.readlines()

    ansList = []
    for qStr in qList:
        ansList.append(predictWord(qStr,fileRoot))
    answerFile = r'F:/NLP/expriment/answer.txt'
    with open(answerFile,'r',encoding = 'utf-8')as f:
        aList = f.readlines()
    cnt = 0
    for index in range(len(aList)):
        print("answer is",aList[index][:-1])
        print(ansList[index])
        for x in ansList[index]:
            if(x == aList[index][:-1]):

                cnt += 1
                break
    print("rate is ",cnt/100)    
if __name__ == '__main__':
    main()