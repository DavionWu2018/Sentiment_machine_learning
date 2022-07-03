

###1 导入数据

import numpy as np
import pandas as pd

data = pd.read_excel('D:\Davion\data_test_train.xlsx')
data.head()


####2 数据预处理 

##去重(建议Excel手动去除空行和重复值)、去除停用词(可单独处理，也可以结合停用词表处理)
##Excel中手动进行，查找重复值，替换特殊符号，去除空格等


####3_1 jieba分词(简单版)

import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()


####3_2 jieba分词(高级版)

import re
import jieba
import jieba.posseg as psg
dic_file = "D:/Davion/add_word_list.txt"
stop_file = "D:/Davion/stopwordlist.txt"

def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file,encoding ='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n','nz','vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    
    word_list = []
    #jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word)<2:
                    find = 1
                    break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)      
    return (" ").join(word_list)

data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()


####4_1 提取特征(CountVectorizer)

from sklearn.feature_extraction.text import CountVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = 'D:\Davion\哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8, 
                       min_df = 3, 
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                       stop_words=frozenset(stopwords))


####4_2 提取特征(TfidfVectorizer)

from sklearn.feature_extraction.text import TfidfVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = 'D:\Davion\哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = TfidfVectorizer(norm='l2', ngram_range=(1, 2))

#features = vect.fit_transform(data.cut_comment)
#print(features.shape)
#print(features[1:5])


####5 划分数据集

##5.1 划分数据集

X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


##5.2 特征展示

test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
test.head()


####6_1 训练模型(朴素贝叶斯)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=0.01)

X_train_vect = vect.fit_transform(X_train)
nb.fit(X_train_vect, y_train)
train_score = nb.score(X_train_vect, y_train)

print(train_score)

####6_2 训练模型(K近邻)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #,weights='distance'

X_train_vect = vect.fit_transform(X_train)
knn.fit(X_train_vect, y_train)
train_score = knn.score(X_train_vect, y_train)

print(train_score)

####6_3 训练模型(支持向量机)

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf',probability = True)

X_train_vect = vect.fit_transform(X_train)
svm.fit(X_train_vect, y_train)
train_score = svm.score(X_train_vect, y_train)

print(train_score)

####6_4 训练模型(Logistic分类)

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()

X_train_vect = vect.fit_transform(X_train)
logit.fit(X_train_vect, y_train)
train_score = logit.score(X_train_vect, y_train)

print(train_score)

####6_5 训练模型(随机森林分类)

from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 8)

X_train_vect = vect.fit_transform(X_train)
ranfor.fit(X_train_vect, y_train)
train_score = ranfor.score(X_train_vect, y_train)

print(train_score)

####6_6 训练模型(决策树分类)

from sklearn import tree
decitree = tree.DecisionTreeClassifier()

X_train_vect = vect.fit_transform(X_train)
decitree.fit(X_train_vect, y_train)
train_score = decitree.score(X_train_vect, y_train)

print(train_score)

####6_7 训练模型(梯度增强决策树分类)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators = 200)

X_train_vect = vect.fit_transform(X_train)
gbdt.fit(X_train_vect, y_train)
train_score = gbdt.score(X_train_vect, y_train)

print(train_score)


####7 测试模型(根据模型替换为xx.score)

X_test_vect = vect.transform(X_test)
print(nb.score(X_test_vect, y_test))


####8 分析数据(根据模型替换为xx_result) 

data = pd.read_excel("D:\Davion\data.xlsx").astype(str)
data.head()

data = pd.read_excel("D:\Davion\data.xlsx").astype(str)
data['cut_comment'] = data.comment.apply(chinese_word_cut)
X=data['cut_comment']
X_vec = vect.transform(X)

nb_result = nb.predict(X_vec)
data['nb_result'] = nb_result

test = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names())
test.head()

data.to_excel("D:\Davion\data_result.xlsx",index=False)

####每次做情感预测都需要重新训练模型，不然会出现dimension mismatch



