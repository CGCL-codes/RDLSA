
import sys
from collections import defaultdict
import os
import re
import jieba
import codecs
import pandas as pd
import numpy as np
import json
# 生成stopword表，需要去除一些否定词和程度词汇
from tqdm import tqdm
from sklearn import metrics

from platform import platform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


stopwords = set()
fr = open('./stopwords.txt', 'r', encoding='utf-8')
for word in fr:
    stopwords.add(word.strip())  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
fr.close()
# 读取否定词文件
not_word_file = open('否定词.txt', 'r+', encoding='utf-8')
not_word_list = not_word_file.readlines()
not_word_file.close()
not_word_list=[]
not_word_list = [w.strip() for w in not_word_list]
# 读取程度副词文件
degree_file = open('程度副词.txt', 'r+')
degree_list = degree_file.readlines()
degree_list = [item.split(',')[0] for item in degree_list]
degree_file.close()
# 生成新的停用词表
with open('stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords:
        if (word not in not_word_list) and (word not in degree_list):
            f.write(word + '\n')


# jieba分词后去除停用词
def seg_word(sentence):
    if dataset == 'train.json':
        seg_list = sentence.strip().split()
    else:
        seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('stopwords.txt', 'r') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, seg_result))


# 找出文本中的情感词、否定词和程度副词
def classify_words(word_list):
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[i] = sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            # 分词结果中在否定词列表中的词
            not_word[i] = -1
        elif word in degree_dict.keys():
            # 分词结果中在程度副词中的词
            degree_word[i] = degree_dict[word]

    # 返回分类结果
    return sen_word, not_word, degree_word


# 计算情感词的分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果
    for i in range(0, len(seg_result)):
        # 如果是情感词
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加一，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，权重取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score


# 计算得分
def sentiment_score(sentence):
    # 1.对文档分词
    try:
        seg_list = seg_word(sentence)
    except Exception as e:
        return 0
    # 2.将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word, not_word, degree_word = classify_words(seg_list)
    # 3.计算得分
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return score


# print("我今天很高兴也非常开心    ", sentiment_score("我今天很高兴也非常开心"))
# print('天灰蒙蒙的，路上有只流浪狗，旁边是破旧不堪的老房子   ', sentiment_score('天灰蒙蒙的，路上有只流浪狗，旁边是破旧不堪的老房子'))
# print('愤怒、悲伤和埋怨解决不了问题    ', sentiment_score('愤怒、悲伤和埋怨解决不了问题'))
# print('要每天都开心快乐    ', sentiment_score('要每天都开心快乐'))
# print('我不喜欢这个世界，我只喜欢你    ', sentiment_score('我不喜欢这个世界，我只喜欢你'))


# 加载文件，导入数据,分词
def loadfile():
    if dataset=='tb':
        neg = pd.read_excel('neg.xls', header=None)
        pos = pd.read_excel('pos.xls', header=None)
        # cw = lambda x: list(jieba.cut(x))
        # pos['words'] = pos[0].apply(cw)
        # neg['words'] = neg[0].apply(cw)

        # print(pos['words'])
        # use 1 for positive sentiment, 0 for negative
        y = list(np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))))
        x = list(np.concatenate((pos[0], neg[0])))
    elif dataset == 'simplifyweibo_4_moods.csv':
        pd_all = pd.read_csv('simplifyweibo_4_moods.csv')
        moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}
        pos = [x for x in pd_all[pd_all.label == 0]['review'].tolist() if len(x) > 3 ]
        neg = [x for x in pd_all[pd_all.label != 0]['review'].tolist() if len(x) > 3 ]
        y = list(np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))))
        x = list(np.concatenate((pos, neg)))
    elif dataset == 'train.json':
        with open('train.json', 'r') as f:
            pd_all = pd.DataFrame(json.loads(f.read()), columns=['review', 'label'])
            moods = {0: 'Null', 1: 'Like', 2: 'Sad', 3: 'Disgust', 4:'Anger', 5:'Happiness'}
            pos = [x for x in pd_all[pd_all.label == 1]['review'].tolist() if len(x) > 3 ] + [x for x in pd_all[pd_all.label == 5 ]['review'].tolist() if len(x) > 3 ]
            neg = [x for x in pd_all[pd_all.label == 2]['review'].tolist() if len(x) > 3 ] +\
                   [x for x in pd_all[pd_all.label == 3]['review'].tolist() if len(x) > 3] +\
                   [x for x in pd_all[pd_all.label == 5]['review'].tolist() if len(x) > 3]
            y = list(np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))))
            x = list(np.concatenate((pos, neg)))
    elif dataset == 'usual':
        emotion_idx = {'surprise': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'happy': 4, 'fear': 5}
        pos = []
        neg = []
        for pathx in [x for x in os.listdir() if x.find('usual') != -1 and x[0]=='u']:
            with open(pathx, 'r') as f:
                cont = json.loads(f.read())
                pos += [x['content'] for x in cont if len(x['content']) > 3 and emotion_idx[x['label']] in [4]]
                neg += [x['content'] for x in cont if len(x['content']) > 3 and emotion_idx[x['label']] in [2,3,5]]
        y = list(np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))))
        x = list(np.concatenate((pos, neg)))
    elif dataset == 'waimai' or dataset=='hotel':
        if dataset == 'waimai':
            pd_all = pd.read_csv('waimai_10k.csv')
        else:
            pd_all = pd.read_csv('hotel.csv')
        pos = [x for x in pd_all[pd_all.label == 1]['review'].tolist()]
        neg = [x for x in pd_all[pd_all.label == 0]['review'].tolist()]
        y = list(np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))))
        x = list(np.concatenate((pos, neg)))
    return x, y # x[:2500] +  x[-2500:], y[:2500] + y[-2500:]

def openTrain(dir, sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            a, b, c = triple
            if a==b:
                continue
            list.append(tuple([a, b, c]))
            num += 1
    return num, list

def addAnt(yes=True):
    sen_file = open('BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取词典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取词典每一行的内容，将其转换成字典对象，key为情感词，value为其对应的权重
    for i in sen_list:
        if len(i.split(' ')) == 2:
            # if len(i.split(' ')[0]) != 1:
            #     continue
            sen_dict[i.split(' ')[0]] = float(i.split(' ')[1])
    sen_file.close()

    if not yes:
        return sen_dict
    else:
        print("添加！")
    with open('../addToSenDict.txt', 'r') as f:
        for i in f.readlines():
            # 此处决定了是添加正向+负例 还是 只添加负例
            # if float(i.split()[1]) > 0:
            #     continue
            try:
                sen_dict[i.split()[0]] = float(i.split()[1])
            except Exception as ee:
                print(ee)
    _, trainList = openTrain('../train.txt')
    _, testList = openTrain('../test.txt')
    _, validList = openTrain('../valid.txt')
    # _, tripleDisc = openTrain('../tripleDisc.txt')
    allTriple = trainList + testList + validList #+ tripleDisc
    rel = {'ANT':-1, 'SYN':1}
    for x in allTriple:
        h,t,r = x[0], x[1], x[2]
        if h in sen_dict and t not in sen_dict:
            sen_dict[t] = sen_dict[h] * rel[r]
        if t in sen_dict and h not in sen_dict:
            sen_dict[h] = sen_dict[t] * rel[r]
    return sen_dict


def test():
    x, y = loadfile()
    pos_right = 0
    pos_wrong = 0
    neg_right = 0
    neg_wrong = 0
    y_test_cls = y
    y_pred_cls = []
    for i in tqdm(range(len(x))):
        xx = x[i]
        yy = y[i]
        if yy == 1:
            if sentiment_score(xx) < 0:
                pos_wrong += 1
                y_pred_cls.append(0)
            else:
                pos_right += 1
                y_pred_cls.append(1)
        else:
            if sentiment_score(xx) < 0:
                y_pred_cls.append(0)
                neg_right += 1
            else:
                y_pred_cls.append(1)
                neg_wrong += 1
    # print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, digits=4, target_names=['Negative','Positive']))
    # print("Confusion Matrix...")
    print(metrics.confusion_matrix(y_test_cls, y_pred_cls))
    # acc = round((pos_right + neg_right) / (pos_right + neg_right + pos_wrong + neg_wrong), 4)
    # pre = round(pos_right / (pos_right + pos_wrong), 4)
    # rec = round(pos_right / (pos_right + neg_wrong), 4)
    # f1 = round(2*pre*rec / (pre + rec), 4)
    # print("Acc:{}, Precision:{}, Recall:{}, F1:{} ".format(acc, pre, rec, f1))
    # print('pos_right:{}, neg_right:{}, pos_wrong:{}, neg_wrong:{}'.format(pos_right, neg_right, pos_wrong, neg_wrong))


def getAllWordInDoc(file='allWord.txt'):
    x, y = loadfile()
    allword = set()
    for i in tqdm(range(len(x))):
        xx=x[i]
        try:
            seg_list = seg_word(xx)
            for ww in seg_list:
                allword.add(ww)
        except Exception as e:
            print(xx)
            continue
    with open(file, 'w') as f:
        for x in allword:
            f.write('{}\n'.format(x))

# patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
#color_list = ['y','g','b', 'k', 'm', 'c','r']


def drawPlot(x, y1, y2,y3,y4, name, en, patterns = ('/','-','\\','-', )):
    plt.style.use("bmh")
    labels = np.arange(len(x))
    fig, ax = plt.subplots()
    width = 0.3
    alpha = 0.5
    ax.bar(labels - width , y1, width / 2, alpha=alpha, color='b', label='Original BosonNLP', hatch=patterns[0])
    ax.bar(labels - width/2 , y2, width / 2, alpha=alpha, color='c', label='Enhanced BosonNLP', hatch=patterns[1])
    ax.bar(labels, y3, width / 2, alpha=alpha, color='k', label='Original HowNet', hatch=patterns[2])
    ax.bar(labels + width/2, y4, width / 2, alpha=alpha, color='r', label='Enhanced HowNet', hatch=patterns[3])

    plt.xticks(x, labels=en)
    # ax.set_ylabel('Accuracy')
    if name=='Accuracy' or name=='F1':
        ax.set_ylim(0.4, 0.8)
    elif name.find('Pos') != -1:
        ax.set_ylim(0.7, 1)
    else:
        ax.set_ylim(0.1, 0.8)


    # ax.set_title('%s'%name, fontsize=16, color='black')
    ax.legend(fontsize=12)
    for x in ax.get_xticklabels():
        x.set_rotation(45)  # 这里是调节横坐标的倾斜度，rotation是度数
    picname = 'enhanced-sentiment-lexicon-%s'%name
    if platform().lower().find('linux') != -1:
        plt.savefig('./{}.png'.format(picname))
    else:
        plt.savefig('/Users/zhangzhaobo/Documents/Papers/Antonym/figure/{}.png'.format(picname))
        plt.show()




if __name__ == '__main__':
    try:
        dataset = sys.argv[1]
    except Exception as e:
        dataset = 'tb'
    try:
        isAddAnt = int(sys.argv[2])
    except Exception as e:
        isAddAnt = 0
        # dataset = 'simplifyweibo_4_moods.csv'
        # dataset = 'train.json'
        # dataset = 'usual'
        # dataset = 'waimai'
        # dataset = 'hotel'
    try:
        picindex = int(sys.argv[3])
    except Exception as e:
        picindex = 3
    trans = {0:False, 1:True}
    sen_dict = addAnt(trans[isAddAnt])
    # sen_dict = addAnt(False)
    # 读取程度副词文件
    degree_file = open('程度副词.txt', 'r+')
    degree_list = degree_file.readlines()
    degree_dict = defaultdict()
    for i in degree_list:
        degree_dict[i.split(',')[0]] = i.split(',')[1]
    # 关闭打开的文件
    not_word_file.close()
    degree_file.close()
    # getAllWordInDoc()
    # getAllWordInDoc('allWord_weibo.txt')
    # getAllWordInDoc('allWord_trainjson.txt')
    # getAllWordInDoc('allWord_usual.txt')
    # getAllWordInDoc('allWord_waimai.txt')
    # getAllWordInDoc('allWord_hotel.txt')

    # test()
    names = ['Taobao', 'Weibo', 'News', 'Sina', 'Waimai', 'Hotel']
    # These results are from TextCNN with our embeddings.
    y1, y2, y3, y4 = [ [[0.6598, 0.6311, 0.6223, 0.785, 0.6191, 0.7552
                ],[0.6624, 0.6316, 0.6234, 0.7872, 0.6173, 0.7595
                ],[0.6069, 0.5711, 0.5125, 0.5103, 0.493, 0.7557
                ],[0.6207, 0.5732, 0.5265, 0.5376, 0.526, 0.758
                ],],[[0.378, 0.337, 0.486, 0.7717, 0.4915, 0.419
                ],[0.3853, 0.3403, 0.4891, 0.7756, 0.4878, 0.4317
                ],[0.3254, 0.2781, 0.2694, 0.3943, 0.2782, 0.4313
                ],[0.3666, 0.3046, 0.3035, 0.4382, 0.334, 0.4595
                ],],[[0.9351, 0.8703, 0.8125, 0.8205, 0.8738, 0.9096
                ],[0.933, 0.8686, 0.8109, 0.8182, 0.8758, 0.91
                ],[0.8818, 0.8094, 0.8519, 0.8211, 0.9217, 0.9047
                ],[0.8689, 0.7917, 0.8377, 0.8039, 0.9093, 0.8952
                ],],[[0.6307, 0.6004, 0.6176, 0.7947, 0.6232, 0.736
                ],[0.6345, 0.6016, 0.619, 0.7966, 0.621, 0.7414
                ],[0.5735, 0.5375, 0.4758, 0.5226, 0.4643, 0.7382
                ],[0.5948, 0.5455, 0.498, 0.5543, 0.51, 0.7438
                ],],][picindex]
    picnames = ['Accuracy', 'Neg Accuracy', 'Pos Accuracy', 'F1']
    drawPlot(x=range(6),
             y1=y1, y2 =y2, y3=y3,y4=y4,
             name=picnames[picindex],
             en=names)

