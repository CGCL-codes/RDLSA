# -*- coding: utf-8 -*-
import os
import platform
import random
import re
import sys
import socket
import time
from random import uniform, sample

import json
from tqdm import tqdm

from sklearn.utils import shuffle
import numpy as np
import jieba
import pandas as pd

from opencc import OpenCC
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import keras as ks
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Activation, Input, Conv1D, concatenate, Add, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
hostname = socket.gethostname()
if str(platform.system()).lower().find('nux') != -1:
    BASE_DIR = '/home/username/HanziNet/'
    HOME_DIR = '/home/username/'
else:
    BASE_DIR = '/Users/username/PycharmProjects/HanziNet/'
    HOME_DIR = '/Users/username/PycharmProjects/'
filtrate = re.compile(u'[^\u4E00-\u9FA5 ]')  # 非中文
sys.path.append(BASE_DIR)


def getGPU():
    if not os.path.exists('gpu_usage.json'):
        os.system('touch gpu_usage.json')
        with open('gpu_usage.json', 'w') as f:
            f.write(json.dumps({
                '0':5,
                '1':0,
                '2':0,
                '3':0
            }, indent=4))
        return '0'
    g = '2'
    usage = None
    with open('gpu_usage.json', 'r') as f:
        usage = json.loads(f.read())
        for gpu, u in usage.items():
            if u==0:
                g = gpu
                break
    usage[gpu] = 5
    with open('gpu_usage.json', 'w') as f:
        f.write(json.dumps(usage, indent=4))
        return g

def releaseGPU(x):
    with open('gpu_usage.json', 'r') as f:
        usage = json.loads(f.read())
        usage[x] = 0
    with open('gpu_usage.json', 'w') as f:
        f.write(json.dumps(usage, indent=4))


GPU = getGPU()
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    print("使用的是{}号GPU".format(GPU))
else:
    os.system('gpustat > gpu_stat_tmp.txt')
    gpus = {
        '0': 5,
        '1': 5,
        '2': 5,
        '3': 5,
    }
    with open('gpu_stat_tmp.txt', 'r') as f:
        for l in f.readlines():
            if l.find('32GB') != -1:
                if l.find('0 / 32510') != -1:
                    GPU = l[l.find('[') + 1: l.find('[') + 2]
                    gpus[GPU] = 0
        gpus[GPU] = 0
        with open('gpu_usage.json', 'w') as f:
            f.write(json.dumps(gpus, indent=4))
    if GPU:
        os.system('rm gpu_stat_tmp.txt')
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    else:
        exit()


def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


def openTrain(dir, sp="\t"):
    # 2023 02 19 添加了从triple discovery得来的三元组们，train + discovery - test
    num = 0
    trilist = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            a, b, c = triple
            if opc.convert(a) == opc.convert(b):
                continue
            trilist.append((a, b, c))
            num += 1
    if dir == './data/train.txt':
        _, newlist = openTrain('./data/test.txt')
        _, newlist1 = openTrain('./data/anti_from_handian_filter.txt')
        _, newlist2 = openTrain('tripleDisc.txt')
        NEWLIST = [x for x in newlist2 if x not in newlist and x not in trilist and x not in newlist1]
        with open('./data/train_td.txt', 'w') as f:
            for x in trilist + NEWLIST:
                f.write('{}\t{}\t{}\n'.format(x[0], x[1],x[2]))
        return num, trilist + NEWLIST
    return num, trilist

def build_dictionary(word_list):
    dictionary = dict()
    cnt = 0
    for w in word_list:
        # if w in dictionary:print('repeat:', w)
        dictionary[w] = cnt
        cnt += 1
    return dictionary


def readVector(vec_file, wf={}):
    global vocab_dim
    # input:  the file of word2vectors
    # output: word dictionay, embedding matrix -- np ndarray
    start = time.time()
    print("\rLoading Vec...", end='')
    f = open(vec_file, 'r', encoding='utf8')
    cnt = 0
    word_list = []
    embeddings = []
    word_size = 0
    embed_dim = 0
    count = 0
    for line in f:
        count += 1
        # if count > 600000:
        #     continue
        data = line.split()
        if cnt == 0:
            word_size = data[0]
            embed_dim = int(data[1])
            # word_list.append('~')
            # embeddings.append([0]*embed_dim)
            if len(data) > 10:
                word_list.append(data[0])
                tmpVec = [float(x) for x in data[-embed_dim:]]
                embeddings.append(tmpVec)
        else:
            if len(data) != embed_dim + 1:
                # print(line, len(data), embed_dim + 1)
                continue
            try:
                if wf and wf[data[0]] < 50:
                    continue
            except Exception as e:
                continue
            word_list.append(data[0])
            # exist_words.add(data[0])
            tmpVec = [float(x) for x in data[-embed_dim:]]
            embeddings.append(tmpVec)
        cnt = cnt + 1
    f.close()
    vocab_dim = int(embed_dim)
    embeddings = np.array(embeddings)
    word_size = len(word_list)
    for i in range(int(word_size)):
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    dict_word = build_dictionary(word_list)
    print("\rLoaded Vec...Time Usgae:%s"%(round(time.time() - start)))
    return word_size, embed_dim, dict_word, embeddings


def all_sememe_need(char2SememeFile='entityWords.txt'):
    needSememes = set()
    with open(char2SememeFile, 'r') as f:
        for line in tqdm(f.readlines()):
            data = line.split()
            sememes = data[2:]
            for x in sememes:
                needSememes.add(x[x.find('|') + 1:])
    return needSememes

# def checkSememeVec(vec='./data/sememe-vec.txt', char2SememeFile='entityWords.txt'):
# you can downlaod some pertrain vector here, https://github.com/Embedding/Chinese-Word-Vectors
def checkSememeVec(vec='/home/username/HanziNet/SenseEmbedding/glove.txt', char2SememeFile='./data/entityWords.txt'):
    sememe_list = ['<PAD>']
    embeddings = [[0]*300]
    needSememes = all_sememe_need()
    with open(vec, 'r') as f:
        cnt = 0
        for line in tqdm(f.readlines()):
            if line.find('nan') != -1:
                continue
            data = line.split()
            if cnt == 0:
                char_size = int(data[0])
                embed_dim = int(data[1])
            else:
                if data[0] not in needSememes:
                    continue
                tmpVec = [float(x) for x in data[-embed_dim:]]
                sememe_list.append(data[0])
                embeddings.append(tmpVec)
            cnt = cnt + 1

    sememe2index = {sememe_list[k]:k for k in range(len(sememe_list))}
    missedSememe = set()
    with open(char2SememeFile, 'r') as f:
        char2Sememe = {}
        for line in tqdm(f.readlines()):
            data = line.split()
            char = data[0]
            length = int(data[1])
            sememes = data[2:]
            newsememes = []
            if length != len(sememes):
                print(line)
            for x in sememes:
                if x[x.find('|') + 1:] not in sememe_list:
                    missedSememe.add(x[x.find('|') + 1:])
                    continue
                newsememes.append(x[x.find('|') + 1:])
            char2Sememe[char] = newsememes
    print(missedSememe)
    return char2Sememe, sememe2index, sememe_list, np.array(embeddings)


def process_rank(rank):
    top1 = 0
    top3 = 0
    top10 = 0
    top20 = 0
    for r in rank:
        if r <= 1:
            top1 += 1
        if r <= 3:
            top3 += 1
        if r <= 10:
            top10 += 1
        if r <= 20:
            top20 += 1
    return top1, top3, top10, top20


class AlignModel(object):
    def __init__(self, entityList, relationList, tripleList):
        self.entityList = entityList  # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.relationList = relationList  # 理由同上
        self.tripleList = tripleList
        self.word_dim = None
        self.word_vocab_size = None
        self.word_vec = None
        self.char_dim = None
        self.char_vocab_size = None
        self.input_length_char = 1
        self.input_sememe_length_char = 10
        self.char_vec = None
        # you can get some character vec from some work, like,  CWE, JWE, 4CWE, they are cited in the paper.
        # self.char_vec_file = BASE_DIR + 'Evaluation/ACW-char-300d.txt'
        self.char_vec_file = BASE_DIR + 'SenseEmbedding/char_exampleAvg_glove_300.txt'
        # self.char_vec_file = BASE_DIR + 'SenseEmbedding/char_senseAvg_glove_300.txt'
        self.sememes, self.char_sememe_vec, self.char2Sememe = sememe_list, embeddings, char2Sememe
        self.char_sememe_vocab_size = len(self.sememes)
        self.s2index = build_dictionary(self.sememes)
        self.char_sememe_dim = 300
        self.char_vocab_size, self.char_dim, self.c2index, self.char_vec = readVector(self.char_vec_file)
        print("Loaded Char2Vec!")
        print("当前参数为：{}, {}, {}, {}".format(self.char_vocab_size, self.char_dim, len(self.c2index), len(self.char_vec)))
        self.antTriples = [x for x in self.tripleList if x[2] == 'ANT' and x[0] in self.c2index and x[1] in self.c2index]
        self.synTriples = [x for x in self.tripleList if x[2] == 'SYN' and x[0] in self.c2index and x[1] in self.c2index]
        print("[SYN Size:{}, ANT Size:{}]".format(len(self.synTriples), len(self.antTriples)))
        self.learningRate = 0.0005
        self.iter = 1000
        self.epoch = 100
        self.batch_size = 1000
        self.margin = 0.1
        self.rel_vec = np.array([[1]* self.char_sememe_dim, [-1]* self.char_sememe_dim])
        self.L1_flag = True
        self.theta = 0.3
        self.rel2index = {'SYN':0, 'ANT':1}


    def calAttention_CS(self, char, sememe):
        """
        tf.tile(
            input,     #输入
            multiples,  #同一维度上复制的次数
            name=None
        )
        """
        char = tf.expand_dims(char, axis=2, name=None, dim=None)
        char_expand = tf.tile(char, (1, 1, self.input_sememe_length_char, 1), name='tile')
        att_before_sum = tf.multiply(char_expand, sememe, name='attention_1')
        att_after_sum = tf.reduce_sum(att_before_sum, axis=-1, keep_dims=True)
        attention_value = tf.exp(att_after_sum)
        all_socre = tf.reduce_sum(attention_value, axis=-2, keep_dims=True, name='all_score')
        all_socre = tf.tile(all_socre, (1, 1, self.input_sememe_length_char, 1), name='att_tile')
        attention = tf.div(attention_value, all_socre)
        sememe_att = tf.reduce_sum(
           tf.multiply(sememe, attention), keepdims=False, axis=-2, name='outputATT')
        return BatchNormalization(axis=-1)(sememe_att)


    def calAttention_NoAtt(self, char, sememe):
        char = tf.expand_dims(char, axis=2, name=None, dim=None)
        char_expand = tf.tile(char, (1, 1, self.input_sememe_length_char, 1), name='tile')
        sememe_att = tf.reduce_mean(
            tf.multiply(sememe, char_expand), keepdims=False, axis=-2)
        return sememe_att

    def calAvgSememe(self, sememe):
        sememe_att = tf.reduce_mean(
            sememe, keepdims=False, axis=-2)
        return sememe_att

    def calAttention_SS(self, sememe1, sememe2):
        attention = tf.exp(tf.multiply(sememe1, sememe2, name='attention'))
        all_socre = tf.reduce_sum(attention, axis=-1, name='all_score')
        # all_socre = tf.squeeze(tf.reduce_sum(attention, axis=-1, name='all_score'), axis=1)
        maxSememe = tf.reduce_max(all_socre, axis=-1, name='argmax', keepdims=True)
        # maxSememe = tf.gather(sememe1, axis=-2, name='maxSememe', indices=maxIndex)
        # atte = tf.squeeze(maxSememe, axis=1, name='outputatt')
        return maxSememe

    def buildNet(self):
        self.pos_h = keras.Input(shape=(self.input_length_char,), name="pos_h")  # Variable-length sequence of ints
        self.pos_t = keras.Input(shape=(self.input_length_char,), name="pos_t")  # Variable-length sequence of ints
        self.pos_rel = keras.Input(shape=(self.input_length_char,), name="pos_r")  # Variable-length sequence of ints
        self.pos_h_sememe = keras.Input(shape=(self.input_length_char, self.input_sememe_length_char), name="pos_h_sememe")  # Variable-length sequence of ints
        self.pos_t_sememe = keras.Input(shape=(self.input_length_char, self.input_sememe_length_char), name="pos_t_sememe")  # Variable-length sequence of ints

        self.pos_h_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_vocab_size, mask_zero=True, weights=[self.char_vec], input_length=self.input_length_char,
                                            trainable=True)(self.pos_h)
        self.pos_t_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_vocab_size, mask_zero=True, weights=[self.char_vec], input_length=self.input_length_char,
                                            trainable=True)(self.pos_t)
        self.pos_rel_feature = Embedding(output_dim=self.char_dim,
                                         input_dim=2,
                                         mask_zero=True,
                                         weights=[self.rel_vec],
                                         input_length=self.input_length_char,
                                         # 12.12发现的，艹了，这个肯定是不可训练啊 我傻了！这个b错误耽误了我起码三四天
                                         trainable=False)(self.pos_rel)
        self.pos_h_sememe_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_sememe_vocab_size, mask_zero=True, weights=[self.char_sememe_vec], input_length=self.input_sememe_length_char,
                                                  trainable=True)(self.pos_h_sememe)
        self.pos_t_sememe_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_sememe_vocab_size, mask_zero=True, weights=[self.char_sememe_vec], input_length=self.input_sememe_length_char,
                                                  trainable=True)(self.pos_t_sememe)
        print("Init POS Embedding")

        # self.pos_h_feature = tf.keras.backend.squeeze(self.pos_h_feature, axis=1)
        # self.pos_t_feature = tf.keras.backend.squeeze(self.pos_t_feature, axis=1)
        # self.pos_h_sememe_feature = tf.keras.backend.squeeze(self.pos_h_sememe_feature, axis=1)
        # self.pos_t_sememe_feature = tf.keras.backend.squeeze(self.pos_t_sememe_feature, axis=1)
        # 12.09 考虑两种CS模式，设置为第一种是自注意，第二种是相互注意
        if ATTENTION_SCHEMA  == 0:
            self.pos_att_tail = self.calAttention_CS(self.pos_t_feature, self.pos_t_sememe_feature)
            self.pos_att_head = self.calAttention_CS(self.pos_h_feature, self.pos_h_sememe_feature)
        elif ATTENTION_SCHEMA == 1:
            self.pos_att_tail = self.calAttention_CS(self.pos_h_feature, self.pos_t_sememe_feature)
            self.pos_att_head = self.calAttention_CS(self.pos_t_feature, self.pos_h_sememe_feature)
        elif ATTENTION_SCHEMA == 2:
            self.pos_att_tail = self.calAttention_NoAtt(self.pos_t_feature, self.pos_t_sememe_feature)
            self.pos_att_head = self.calAttention_NoAtt(self.pos_h_feature, self.pos_h_sememe_feature)
        elif ATTENTION_SCHEMA == 3:
            self.pos_att_tail = self.calAttention_NoAtt(self.pos_t_feature, self.pos_h_sememe_feature)
            self.pos_att_head = self.calAttention_NoAtt(self.pos_h_feature, self.pos_t_sememe_feature)
        elif ATTENTION_SCHEMA == 4:
            self.pos_att_tail = self.calAvgSememe(self.pos_t_sememe_feature)
            self.pos_att_head = self.calAvgSememe(self.pos_h_sememe_feature)
        elif ATTENTION_SCHEMA == 5:
            self.pos_att_tail = self.calAvgSememe(self.pos_h_sememe_feature)
            self.pos_att_head = self.calAvgSememe(self.pos_t_sememe_feature)
        elif ATTENTION_SCHEMA == 6:
            self.pos_att_tail = self.calAttention_SS(self.pos_t_sememe_feature, self.pos_h_sememe_feature)
            self.pos_att_head = self.calAttention_SS(self.pos_h_sememe_feature, self.pos_t_sememe_feature)
        self.pos_head = tf.multiply(self.pos_att_head, self.pos_h_feature)
        self.pos_tail = tf.multiply(self.pos_att_tail, self.pos_t_feature)

        self.neg_h = keras.Input(shape=(self.input_length_char,), name="neg_h")  # Variable-length sequence of ints
        self.neg_t = keras.Input(shape=(self.input_length_char,), name="neg_t")  # Variable-length sequence of ints
        self.neg_rel = keras.Input(shape=(self.input_length_char,), name="neg_r")  # Variable-length sequence of ints
        self.neg_h_sememe = keras.Input(shape=(self.input_length_char, self.input_sememe_length_char), name="neg_h_sememe")  # Variable-length sequence of ints
        self.neg_t_sememe = keras.Input(shape=(self.input_length_char, self.input_sememe_length_char), name="neg_t_sememe")  # Variable-length sequence of ints

        self.neg_h_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_vocab_size, mask_zero=True, weights=[self.char_vec], input_length=self.input_length_char,
                                            trainable=True)(self.neg_h)
        self.neg_t_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_vocab_size, mask_zero=True, weights=[self.char_vec], input_length=self.input_length_char,
                                            trainable=True)(self.neg_t)
        self.neg_rel_feature = Embedding(output_dim=self.char_dim,
                                         input_dim=2,
                                         mask_zero=True,
                                         weights=[self.rel_vec],
                                         input_length=self.input_length_char,
                                         trainable=False)(self.neg_rel)
        self.neg_h_sememe_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_sememe_vocab_size, mask_zero=True, weights=[self.char_sememe_vec], input_length=self.input_sememe_length_char,
                                                  # trainable=False))
                                                  trainable=True)(self.neg_h_sememe)
        self.neg_t_sememe_feature = Embedding(output_dim=self.char_dim, input_dim=self.char_sememe_vocab_size, mask_zero=True, weights=[self.char_sememe_vec], input_length=self.input_sememe_length_char,
                                                  # trainable=False))
                                                  trainable=True)(self.neg_t_sememe)
        print("Init NEG Embedding")

        if ATTENTION_SCHEMA == 0:
            # print("字符与自己的sememe相互关注")
            self.neg_att_tail = self.calAttention_CS(self.neg_t_feature, self.neg_t_sememe_feature)
            self.neg_att_head = self.calAttention_CS(self.neg_h_feature, self.neg_h_sememe_feature)
        elif ATTENTION_SCHEMA == 1:
            # print("字符与对方的sememe相互关注")
            self.neg_att_tail = self.calAttention_CS(self.neg_h_feature, self.neg_t_sememe_feature)
            self.neg_att_head = self.calAttention_CS(self.neg_t_feature, self.neg_h_sememe_feature)
        elif ATTENTION_SCHEMA == 2:
            # print("字符与自己的sememe直接乘作为关注")
            self.neg_att_tail = self.calAttention_NoAtt(self.neg_t_feature, self.neg_t_sememe_feature)
            self.neg_att_head = self.calAttention_NoAtt(self.neg_h_feature, self.neg_h_sememe_feature)
        elif ATTENTION_SCHEMA == 3:
            # print("字符与对方的sememe直接乘作为关注")
            self.neg_att_tail = self.calAttention_NoAtt(self.neg_t_feature, self.neg_h_sememe_feature)
            self.neg_att_head = self.calAttention_NoAtt(self.neg_h_feature, self.neg_t_sememe_feature)
        elif ATTENTION_SCHEMA == 4:
            # print("字符与自己的sememe的mean直接乘法，无关注")
            self.neg_att_tail = self.calAvgSememe(self.neg_t_sememe_feature)
            self.neg_att_head = self.calAvgSememe(self.neg_h_sememe_feature)
        elif ATTENTION_SCHEMA == 5:
            # print("字符与对方的sememe的mean直接乘法，无关注")
            self.neg_att_tail = self.calAvgSememe(self.neg_h_sememe_feature)
            self.neg_att_head = self.calAvgSememe(self.neg_t_sememe_feature)
        elif ATTENTION_SCHEMA == 6:
            # print("字符的sememe与对方的sememe的做相互关注，然后取最大")
            self.neg_att_tail = self.calAttention_SS(self.neg_t_sememe_feature, self.neg_h_sememe_feature)
            self.neg_att_head = self.calAttention_SS(self.neg_h_sememe_feature, self.neg_t_sememe_feature)
        self.neg_head = tf.multiply(self.neg_att_head, self.neg_h_feature)
        self.neg_tail = tf.multiply(self.neg_att_tail, self.neg_t_feature)
        self.neg_head = tf.multiply(self.neg_att_head, self.neg_h_feature)
        self.neg_tail = tf.multiply(self.neg_att_tail, self.neg_t_feature)

        if self.L1_flag:
            self.pos = tf.reduce_sum(abs(tf.multiply(self.pos_head, self.pos_rel_feature) + self.pos_tail), 1, keepdims=True)
            self.neg = tf.reduce_sum(abs(tf.multiply(self.neg_head, self.neg_rel_feature) + self.neg_tail), 1, keepdims=True)
            self.predict = self.pos
        else:
            self.pos = tf.reduce_sum(abs(tf.multiply(self.pos_head, self.pos_rel_feature) + self.pos_tail) ** 2 , 1, keepdims=True)
            self.neg = tf.reduce_sum(abs(tf.multiply(self.neg_head, self.neg_rel_feature) + self.neg_tail) ** 2 , 1, keepdims=True)
            self.predict = self.pos

        self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + self.margin, 0), -1, keepdims=False)


    def buildModel(self):
        # 添加正则化器
        self.model = keras.Model(inputs=[self.pos_h, self.pos_t, self.pos_rel, self.pos_h_sememe, self.pos_t_sememe,
                                         self.neg_h, self.neg_t, self.neg_rel, self.neg_h_sememe, self.neg_t_sememe,],
                            outputs=[self.loss], )
        self.model.summary()
        # keras.utils.plot_model(self.model, "AntonymModel.png")  # , show_shapes=True)
        return self.model

    def buildMiddleModel(self):
        # 添加正则化器
        self.middleModel = keras.Model(inputs=[self.pos_h, self.pos_t, self.pos_rel, self.pos_h_sememe, self.pos_t_sememe,
                                         self.neg_h, self.neg_t, self.neg_rel, self.neg_h_sememe, self.neg_t_sememe,],
                             # 12.09  这里还要反复测试到底哪个位置导出为最佳，以及在何种模式下
                            outputs=[self.pos_h_feature, self.pos_t_feature, self.pos_head, self.pos_tail, self.neg_head, self.neg_tail], )
        return self.middleModel

    def getSample(self, size):
        reSample = []
        for i in range(size):
            if random.randint(0, 100) / 100 < self.theta:
                aa = sample(self.antTriples, 1)[0]
                reSample.append(aa)
            else:
                aa = sample(self.synTriples, 1)[0]
                reSample.append(aa)
        return reSample


    def dataGen(self, batch_size, needChar=[], test_pair=()):
        Sbatch = self.getSample(batch_size)
        if len(needChar) != 0:
            Sbatch = [(a, needChar[random.randint(0, len(needChar) - 1 )], 'ANT') for a in needChar]
        if len(test_pair) > 0:
            if len(test_pair) == 3:
                Sbatch = [(test_pair[0], test_pair[1], test_pair[2])]
            else:
                Sbatch = test_pair
        Tbatch = []
        pos_h = []
        pos_t = []
        pos_r = []
        pos_h_sememe = []
        pos_t_sememe = []
        neg_h = []
        neg_t = []
        neg_r = []
        neg_h_sememe = []
        neg_t_sememe = []

        for sbatch in Sbatch:
            h, t, r = sbatch
            # if opc.convert(h) in self.c2index and opc.convert(h) in self.char2Sememe:
            #     h = opc.convert(h)
            # if opc.convert(t) in self.c2index  and opc.convert(t) in self.char2Sememe:
            #     t = opc.convert(t)

            pos_h.append(self.c2index[h])
            pos_t.append(self.c2index[t])
            pos_r.append(self.rel2index[r])
            try:
                pos_h_sememe.append([self.s2index[x] for x in char2Sememe[h]])
            except Exception as e:
                pos_h_sememe.append([])
            try:
                pos_t_sememe.append([self.s2index[x] for x in char2Sememe[t]])
            except Exception as e:
                pos_t_sememe.append([])

            if len(test_pair) <= 3:
                tbatch = self.getCorruptedTriplet(sbatch)
                while tbatch in Tbatch:
                    tbatch = self.getCorruptedTriplet(sbatch)
            else:
                tbatch = sbatch
            h, t, r = tbatch
            # if opc.convert(h) in self.c2index and opc.convert(h) in self.char2Sememe:
            #     h = opc.convert(h)
            # if opc.convert(t) in self.c2index and opc.convert(t) in self.char2Sememe:
            #     t = opc.convert(t)
            neg_h.append(self.c2index[h])
            neg_t.append(self.c2index[t])
            neg_r.append(self.rel2index[r])
            if len(needChar) != 0:
                neg_h_sememe.append([])
                neg_t_sememe.append([])
            else:
                try:
                    neg_h_sememe.append([self.s2index[x]
                                     for x in char2Sememe[h]])
                except Exception as e:
                    neg_h_sememe.append([])
                try:
                    neg_t_sememe.append([self.s2index[x]
                                     for x in char2Sememe[t]])
                except Exception as e:
                    neg_t_sememe.append([])

        pos_h_sememe = tf.keras.preprocessing.sequence.pad_sequences(pos_h_sememe, value=0, padding='post', maxlen=self.input_sememe_length_char)
        pos_t_sememe = tf.keras.preprocessing.sequence.pad_sequences(pos_t_sememe, value=0, padding='post', maxlen=self.input_sememe_length_char)
        neg_h_sememe = tf.keras.preprocessing.sequence.pad_sequences(neg_h_sememe, value=0, padding='post', maxlen=self.input_sememe_length_char)
        neg_t_sememe = tf.keras.preprocessing.sequence.pad_sequences(neg_t_sememe, value=0, padding='post', maxlen=self.input_sememe_length_char)
        return np.array(pos_h), np.array(pos_t), np.array(pos_r), np.array(pos_h_sememe), np.array(pos_t_sememe), np.array(neg_h), np.array(neg_t), np.array(neg_r), np.array(neg_h_sememe), np.array(neg_t_sememe)


    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)
        if len(triplet) != 3:
            print(triplet)
        entityTemp = ''
        if i < 0:  # 小于0，打坏三元组的第一项
            while True:
                entityTemp = sample(self.entityList, 1)[0]
                # 在这里添加了一个要在self.c2index里面，因为经常会出现keyerror
                if entityTemp != triplet[0] and entityTemp in self.c2index:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        else:  # 大于等于0，打坏三元组的第二项
            while True:
                entityTemp = sample(self.entityList, 1)[0]
                if entityTemp != triplet[1] and entityTemp in self.c2index:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2])
        while corruptedTriplet in self.tripleList:
            corruptedTriplet = self.getCorruptedTriplet(triplet)
        return corruptedTriplet


    def getCharEmbedding(self, tmpName=''):
        characters = list(self.c2index.keys())
        # print("字符一共有:{}".format(len(set(characters))))
        pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe = self.dataGen(self.batch_size, test_pair=[(a, a ,'ANT') for a in characters])

        pos_h = np.expand_dims(np.array(pos_h), axis=1)
        pos_t = np.expand_dims(np.array(pos_t), axis=1)
        pos_r = np.expand_dims(np.array(pos_r), axis=1)
        pos_h_sememe = np.expand_dims(np.array(pos_h_sememe), axis=1)
        pos_t_sememe = np.expand_dims(np.array(pos_t_sememe), axis=1)
        #
        neg_h = np.expand_dims(np.array(neg_h), axis=1)
        neg_t = np.expand_dims(np.array(neg_t), axis=1)
        neg_r = np.expand_dims(np.array(neg_r), axis=1)
        neg_h_sememe = np.expand_dims(np.array(neg_h_sememe), axis=1)
        neg_t_sememe = np.expand_dims(np.array(neg_t_sememe), axis=1)
        # 20230213 今天SS版本一直报错，先print各个shape，然后再去对应的调整吧。
        for x in [pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe]:
            print(x.shape, end='__|__')
        chars_emb, _, _, _, _, _  = self.middleModel.predict({
                    'pos_h':pos_h,
                    'pos_t':pos_t,
                    'pos_r':pos_r,
                    'pos_h_sememe':pos_h_sememe,
                    'pos_t_sememe':pos_t_sememe,
                    'neg_h': neg_h,
                    'neg_t': neg_t,
                    'neg_r': neg_r,
                    'neg_h_sememe': neg_h_sememe,
                    'neg_t_sememe': neg_t_sememe,
                })
        # print(chars_emb.shape, pos_h.shape)
        chars_emb = np.squeeze(chars_emb, axis=1)
        # chars_emb_after = np.squeeze(chars_emb_after, axis=1)
        pos_h = np.squeeze(pos_h, axis=1)
        # print(chars_emb.shape,chars_emb_after.shape, pos_h.shape)
        self.charEmbedding = {}
        for char, emb in zip(pos_h, chars_emb):
            # emb = np.array(emb) / np.linalg.norm(emb)
            if characters[char] not in entityList:
                continue
            self.charEmbedding[characters[char]] = emb
        print("应有字符：{}, 抽取字符：{}".format(len(characters), len(self.charEmbedding)))
        if len(tmpName) > 0:
            with open(tmpName, 'w') as output:
                output.write("{}\t{}\n".format(len(characters), self.char_dim))
                for k, v in self.charEmbedding.items():
                    output.write("{}\t{}\n".format(k, "\t".join([str(vv) for vv in v])))
        else:
            with open('./vec/entityVector{}_{}_{}_{}_{}_cs.txt'.format(self.iter, self.epoch, self.batch_size, self.learningRate, ATTENTION_SCHEMA), 'w') as output:
                output.write("{}\t{}\n".format(len(characters), self.char_dim))
                for k, v in self.charEmbedding.items():
                    output.write("{}\t{}\n".format(k, "\t".join([str(vv) for vv in v])))
            os.system("cp {} entityVector_cs_{}.txt".format('./vec/entityVector{}_{}_{}_{}_{}_cs.txt'.format(self.iter, self.epoch, self.batch_size, self.learningRate, ATTENTION_SCHEMA), ATTENTION_SCHEMA))

            self.charEmbedding = {}
            # for char, emb in zip(pos_h, chars_emb_after):
            #     # emb = np.array(emb) / np.linalg.norm(emb)
            #     if characters[char] not in entityList:
            #         continue
            #     self.charEmbedding[characters[char]] = emb
            with open('./vec/entityVector{}_{}_{}_{}_{}_cs_after.txt'.format(self.iter, self.epoch, self.batch_size, self.learningRate, ATTENTION_SCHEMA), 'w') as output:
                output.write("{}\t{}\n".format(len(characters), self.char_dim))
                for k, v in self.charEmbedding.items():
                    output.write("{}\t{}\n".format(k, "\t".join([str(vv) for vv in v])))
            os.system("cp {} entityVector_cs_after.txt".format(
                './vec/entityVector{}_{}_{}_{}_{}_cs_after.txt'.format(self.iter, self.epoch, self.batch_size, self.learningRate, ATTENTION_SCHEMA)))


    def train(self, lossFunc='mae'):

        self.model = self.buildModel()
        self.middleModel = self.buildMiddleModel()
        # optimizer = RMSprop(0.001)
        optimizer = Adam(lr=self.learningRate)#.minimize()
        lf = {
            'mse':tf.keras.losses.mse,
            'mae':tf.keras.losses.mae,
            'mape':tf.keras.losses.mape,
            'hinge':tf.keras.losses.hinge,
            'squared_hinge':tf.keras.losses.squared_hinge
        }
        self.model.compile(
            loss=lf[lossFunc],
            optimizer=optimizer,
            # loss_weights=[self.loss],
        )
        maxAcc_1 = 0
        maxAcc_2 = 0
        for epoch in tqdm(range(self.iter)):
            pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe = self.dataGen(self.batch_size)

            pos_h = np.expand_dims(np.array(pos_h), axis=1)
            pos_t = np.expand_dims(np.array(pos_t), axis=1)
            pos_r = np.expand_dims(np.array(pos_r), axis=1)
            pos_h_sememe = np.expand_dims(np.array(pos_h_sememe), axis=1)
            pos_t_sememe = np.expand_dims(np.array(pos_t_sememe), axis=1)
            #
            neg_h = np.expand_dims(np.array(neg_h), axis=1)
            neg_t = np.expand_dims(np.array(neg_t), axis=1)
            neg_r = np.expand_dims(np.array(neg_r), axis=1)
            neg_h_sememe = np.expand_dims(np.array(neg_h_sememe), axis=1)
            neg_t_sememe = np.expand_dims(np.array(neg_t_sememe), axis=1)

            # z = np.array([random.randint(0,50)/1000 for _ in range(self.batch_size)])
            z = np.array([0 for _ in range(self.batch_size)])
            if epoch==0:
                for x in [pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe]:
                    print(x.shape, end='')
            self.model.fit(
                x={
                    'pos_h':pos_h,
                    'pos_t':pos_t,
                    'pos_r':pos_r,
                    'pos_h_sememe':pos_h_sememe,
                    'pos_t_sememe':pos_t_sememe,
                    'neg_h': neg_h,
                    'neg_t': neg_t,
                    'neg_r': neg_r,
                    'neg_h_sememe': neg_h_sememe,
                    'neg_t_sememe': neg_t_sememe,
                },
                y=z,
                epochs=self.epoch,
                batch_size=self.batch_size,
                # callbacks=[reduce_lr, early_stopping],
                verbose=0,
            )
            gap = self.iter//4
            if epoch > 0 and epoch % gap == 0:
                print("[epoch] %s 当前loss："%epoch, end='')
                loss = self.model.predict(x={
                    'pos_h':pos_h,
                    'pos_t':pos_t,
                    'pos_r':pos_r,
                    'pos_h_sememe':pos_h_sememe,
                    'pos_t_sememe':pos_t_sememe,
                    'neg_h': neg_h,
                    'neg_t': neg_t,
                    'neg_r': neg_r,
                    'neg_h_sememe': neg_h_sememe,
                    'neg_t_sememe': neg_t_sememe,})
                loss = np.array(loss).sum(axis=0)[0]
                print(loss,)
                name = 'entityVector_%s.txt'%("_".join([str(random.randint(1,9)) for _ in range(20)]))
                self.getCharEmbedding(name)
                os.system('python Evaluation.py 1 %s; rm %s'%(name, name))
                acc1 = self.tripleClassification('./data/valid.txt')
                acc2 = self.tripleClassification('./data/anti_from_handian_filter.txt')
                self.link_prediction('./data/valid.txt')
                self.link_prediction('./data/anti_from_handian_filter.txt')
                if acc1 > maxAcc_1:
                    maxAcc_1 = acc1
                if acc2 > maxAcc_2:
                    maxAcc_2 = acc2

    def tripleClassification(self, testFile='./data/test.txt'):
        tris = []
        with open(testFile, 'r') as f:
            for line in f.readlines():
                h, t, r = line.strip().split()
                # if r == 'SYN':
                #     continue
                if h not in self.c2index or t not in self.c2index:
                    continue
                # tris.append((opc.convert(h), opc.convert(t), r))
                # tris.append((opc.convert(t), opc.convert(h), r))
                tris.append((h,t,r))
                tris.append((t,h,r))
        right = 0
        wrong = 0
        for tri in tqdm(tris):
            pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe = self.dataGen(self.batch_size, test_pair=tri)
            pos_h = np.expand_dims(np.array(pos_h), axis=1)
            pos_t = np.expand_dims(np.array(pos_t), axis=1)
            pos_r = np.expand_dims(np.array(pos_r), axis=1)
            pos_h_sememe = np.expand_dims(np.array(pos_h_sememe), axis=1)
            pos_t_sememe = np.expand_dims(np.array(pos_t_sememe), axis=1)
            #
            neg_h = np.expand_dims(np.array(neg_h), axis=1)
            neg_t = np.expand_dims(np.array(neg_t), axis=1)
            neg_r = np.expand_dims(np.array(neg_r), axis=1)
            neg_h_sememe = np.expand_dims(np.array(neg_h_sememe), axis=1)
            neg_t_sememe = np.expand_dims(np.array(neg_t_sememe), axis=1)
            _, _, ph, pt, nh, nt = self.middleModel.predict(x={
                    'pos_h':pos_h,
                    'pos_t':pos_t,
                    'pos_r':pos_r,
                    'pos_h_sememe':pos_h_sememe,
                    'pos_t_sememe':pos_t_sememe,
                    'neg_h': neg_h,
                    'neg_t': neg_t,
                    'neg_r': neg_r,
                    'neg_h_sememe': neg_h_sememe,
                    'neg_t_sememe': neg_t_sememe,})
            pos_r = np.squeeze(pos_r, axis=1)[0]
            if pos_r == 0:
                if abs(np.sum(ph + pt)) - abs(np.sum(nh + nt)) < 0:
                    right += 1
                else:
                    wrong += 1
            else:
                if abs(np.sum(ph - pt)) - abs(np.sum(nh - nt)) < 0:
                    right += 1
                else:
                    wrong += 1
        print("准确率: {}/{} = {}".format(right, (right + wrong), round(right / (right + wrong), 4) ) )
        return round(right / (right + wrong), 4)

    def link_prediction(self, testFile='./data/test.txt'):
        tris = []
        with open(testFile, 'r') as f:
            for line in f.readlines():
                h, t, r = line.strip().split()
                if r=='SYN':
                    continue
                if h not in self.c2index or t not in self.c2index:
                    continue
                # tris.append((opc.convert(h), opc.convert(t), r))
                # tris.append((opc.convert(t), opc.convert(h), r))
                tris.append((h,t,r))
                tris.append((t,h,r))
        ranks = []
        for tri in tqdm(tris):
            test_pairs = []
            test_pairs.append(tri)
            for char in self.entityList:
                if char == tri[0] or char == tri[1] or char not in self.c2index:
                    continue
                test_pairs.append((tri[0], char, tri[2]))
            start = time.time()
            pos_h, pos_t, pos_r, pos_h_sememe, pos_t_sememe, neg_h, neg_t, neg_r, neg_h_sememe, neg_t_sememe = self.dataGen(self.batch_size, test_pair=test_pairs)
            pos_h = np.expand_dims(np.array(pos_h), axis=1)
            pos_t = np.expand_dims(np.array(pos_t), axis=1)
            pos_r = np.expand_dims(np.array(pos_r), axis=1)
            pos_h_sememe = np.expand_dims(np.array(pos_h_sememe), axis=1)
            pos_t_sememe = np.expand_dims(np.array(pos_t_sememe), axis=1)
            #
            neg_h = np.expand_dims(np.array(neg_h), axis=1)
            neg_t = np.expand_dims(np.array(neg_t), axis=1)
            neg_r = np.expand_dims(np.array(neg_r), axis=1)
            neg_h_sememe = np.expand_dims(np.array(neg_h_sememe), axis=1)
            neg_t_sememe = np.expand_dims(np.array(neg_t_sememe), axis=1)
            # print('数据生成时间:{}'.format((time.time()-start)))
            # start = time.time()
            ph, pt, _, _, nh, nt = self.middleModel.predict(x={
                    'pos_h':pos_h,
                    'pos_t':pos_t,
                    'pos_r':pos_r,
                    'pos_h_sememe':pos_h_sememe,
                    'pos_t_sememe':pos_t_sememe,
                    'neg_h': neg_h,
                    'neg_t': neg_t,
                    'neg_r': neg_r,
                    'neg_h_sememe': neg_h_sememe,
                    'neg_t_sememe': neg_t_sememe,})
            pos_r = np.squeeze(pos_r, axis=1)[0]
            ph, pt = np.squeeze(np.array(ph), axis=1), np.squeeze(np.array(pt), axis=1)
            if pos_r == 'ANT' or pos_r==1:
                scores = np.sum(ph + pt, axis=-1)
            else:
                scores = np.sum(ph - pt, axis=-1)
            target_vsim = scores[0]
            sortedVsim = sorted(scores, reverse=False)
            rank = sortedVsim.index(target_vsim)
            ranks.append(rank)
            # print('数据处理时间:{}'.format((time.time()-start)))
        head_1, head_3, head_10, head_20 = process_rank(ranks)
        if len(ranks) < 150:
            print(ranks)
        print("head: top1:{}, top5:{}, top10:{}, MRR:{}, MR:{}/ {}".format(head_1, head_3, head_10,
                                                                           round(np.mean([1 / x for x in ranks]),
                                                                                 4), round(np.mean(ranks)),
                                                                           len(ranks)))

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("\n Parameter Annotation \n "
              "lossf: str - Loss function type, default is 'mae'  \n"
              "ite: int - Number of training iterations, default is 1000 \n"
              "lr: float - Learning rate, default is 0.0005 \n "
              "ATTENTION_SCHEMA: int - Attention schema type, default is 0(CS), left is 2(Avg), 6(SS)")
        print('Following command is a example for runing RDLSA:'
              'python TrainAntonymAttention.py mae 500 0.000001 0')
        exit(0)


    ATTENTION_SCHEMA = 0
    time.sleep(1)
    try:
        lossf = sys.argv[1]
    except Exception as e:
        lossf = 'mae'
    try:
        ite = sys.argv[2]
    except Exception as e:
        ite = 1000
    try:
        lr = sys.argv[3]
    except Exception as e:
        lr = 0.0005
    try:
        ATTENTION_SCHEMA = int(sys.argv[4])
    except Exception as e:
        ATTENTION_SCHEMA = 0
    opc = OpenCC('t2s')
    seed = str(random.randint(0, 10000)) + '-' + str(random.randint(0,10000))
    os.system('cp {} backup/{}_{}'.format(sys.argv[0], seed, sys.argv[0]))
    dirEntity = "./data/entity2id.txt"
    # 读取实体id 以及id所对应的实体列表
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirRelation = "./data/relation2id.txt"
    # 读取关系id 以及id所对应的关系列表
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    dirTrain = "./data/train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    char2Sememe, sememe2index, sememe_list, embeddings = checkSememeVec()
    # print('损失函数:{}，迭代次数:{}, 学习率:{}, 激活函数:{}'.format(lossf, ite, lr, acti))
    model = AlignModel(entityList, relationList, tripleList)
    model.learningRate = float(lr)
    model.iter = int(ite)
    # print(char_emb_file)
    # model.char_vec_file = BASE_DIR + char_emb_file
    model.buildNet()
    print('build Net success!')
    model.train(lossf)
    print('train success!')
    print('损失函数:{}，迭代次数:{}, 学习率:{}, '.format(lossf, ite, lr))
    print('使用的字符嵌入为', model.char_vec_file)
    releaseGPU(GPU)
    model.getCharEmbedding()

