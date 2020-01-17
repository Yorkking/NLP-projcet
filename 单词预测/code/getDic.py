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