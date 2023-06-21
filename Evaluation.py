import random
from platform import platform

import numpy as np
from tqdm import tqdm
import sys
import os
from get_pos_word import pos_all_word
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


L1_flag = 'L1'



def openEntitySememe(dir='./data/entityWords.txt'):
    ent2sememe = {}
    with open(dir, 'r') as f:
        for line in f.readlines():
            x = line.strip().split()
            ent = x[0]
            if len(x) > 3 and len(x[2:]) == int(x[1]):
                sememe = x[2:]
            else:
                try:
                    sememe = x[2].split()
                except Exception as e:
                    # print(line)
                    sememe = []
            ent2sememe[ent] = sememe
    return ent2sememe

def multiwise(x, y):
    if len(x) != len(y):
        print("长度不对")
        return
    z = []
    for x1 in x:
        for y1 in y:
            z.append(x1*y1)
    return z


def calc_sim(e1, e2, rel):
    if rel=='SYN':
        L1_flag='cos'
    else:
        L1_flag = 'L1'
    if L1_flag == 'cos':
        return abs(e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    elif L1_flag == 'L1':
        # d = ((e1 - e2) * (e1 - e2)).sum()
        if rel=='ANT':
            d = np.linalg.norm(e1 + e2)
        else:
            d = np.linalg.norm(e1 - e2)
        return d
    elif L1_flag == 'L2':
        if rel=='ANT':
            return pow((e1 + e2).sum(), 2)
        else:
            return pow((e1 - e2).sum(), 2)


def read_file(filename, triple=False):
    if not triple:
        ent2id = {}
        id2ent = {}
    else:
        tris = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not triple:
                d, idx = line.strip().split()
                ent2id[d] = idx
                id2ent[idx] = d
            else:
                h, t, r = line.strip().split()
                tris.append((h,r,t))
    if triple:
        return tris
    return ent2id, id2ent

def readVector(filename, charlist=[]):
    # print("Loading Vec:", filename)
    word_list = []
    embeddings = []
    char_emb = {}
    count = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.replace('[', '').replace(']', '').replace(',', '')
            data = line.strip().split()
            if count == 0:
                count += 1
                continue
            if len(charlist) > 0 and data[0] not in charlist:
                continue

            if len(data) != 2:
                # print(len(data))
                word_list.append(data[0])
                tmpVec = [float(x) for x in data[1:]]
            else:
                print("妈的？智障")
                word_list.append(data[0])
                tmpVec = [float(x) for x in data[1].split()]
            embeddings.append(tmpVec)
        embeddings = np.array(embeddings)
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
            char_emb[word_list[i]] = embeddings[i]
    return char_emb


def cal_top(tri, char_emb, mask_pos):
    h, r, t = tri
    h_emb = char_emb[h]
    t_emb = char_emb[t]

    if mask_pos in ['tail', 'head']:
        vsim = []
        target_vsim = -1
        if mask_pos == 'head':
            target = h
        else:
            target = t
        for k, v in char_emb.items():
            if mask_pos=='head':
                sim = calc_sim(v, t_emb, r)
            else:
                sim = calc_sim(h_emb, v, r)
            # print(sim)
            if k==target:
                target_vsim = sim
            vsim.append(sim)
        # if r == 'SYN':
        # sortedVsim = sorted(vsim)
        # else:
        sortedVsim = sorted(vsim, reverse=True)
        # print(sortedVsim)
        rank = sortedVsim.index(target_vsim)
        return rank

def link_prediction(arg, emb, dataset='./data/test.txt'):
    existChar = list(readVector(emb, entityList).keys())
    test_triples = read_file(dataset, True)
    char_emb = readVector(emb, existChar)
    ranks_head = []
    ranks_tail = []
    for tri in tqdm(test_triples):
        if arg != 0 and tri[1] == 'SYN':
            continue
        ranks_head.append(cal_top(tri, char_emb, 'head'))
        ranks_tail.append(cal_top(tri, char_emb, 'tail'))
        # break
        # ranks_rel.append(cal_top(tri, char_emb, rel_emb, 'rel'))
    return ranks_head, ranks_tail


def process_rank(rank):
    print(rank)
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

def openTrain(dir, sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list

def getRandomTriple(existChar, h, r, t, head=True):
    if head:
        h_ = existChar[(existChar.index(h) + len(existChar)//2) % len(existChar)] # random.randint(0, len(existChar))) % len(existChar)]
        # print(h,h_)
        neg_tri = (h_, r, t)
        while neg_tri in allTriples:
            h_ = existChar[(existChar.index(h_) + 1) % len(existChar)]
            # h_ = existChar[(existChar.index(h) + random.randint(0, len(existChar))) % len(existChar)]
            neg_tri = (h_, r, t)
    else:
        t_ = existChar[(existChar.index(t) + len(existChar)//2) % len(existChar)] # random.randint(0, len(existChar))) % len(existChar)]
        neg_tri = (h, r, t_)
        while neg_tri in allTriples:
            t_ = existChar[(existChar.index(t_) + 1) % len(existChar)]
            # t_ = existChar[(existChar.index(t) + random.randint(0, len(existChar))) % len(existChar)]
            neg_tri = (h, r, t_)
    return neg_tri


def distanceL(h, t, r):
    h = np.array(h)
    t = np.array(t)
    # return h.dot(t) / (np.linalg.norm(h) * np.linalg.norm(t))
    return calc_sim(h, t, r)


def triple_classification(arg, emb_file, head=True, dataset='test.txt'):
    test_triples = read_file(dataset, True)
    char_emb = readVector(emb_file, entityList)
    existChar = list(char_emb.keys())
    right_times = 0
    wrong_times = 0
    newpair = []
    for tri in tqdm(set(test_triples)):
        h, r, t = tri
        if arg != 0 and r == 'SYN':
            continue
        neg_tri = getRandomTriple(existChar, h, r, t, head)
        while neg_tri in test_triples:
            neg_tri = getRandomTriple(existChar, h, r, t)
        pos_score = distanceL(char_emb[h], char_emb[t], r)
        neg_score = distanceL(char_emb[neg_tri[0]], char_emb[neg_tri[2]], r)
        if head:
            www = h
        else:
            www = t
        if pos_score > neg_score:
            try:
                try:
                    sese = len(char2Sememe[www])
                except Exception as e:
                    sese = 99
                if len(dataset) < 10:
                    ttt = pos_name_1[word2pos_1[www]]
                    sememeNum2Char_1[sese] += 1
                    sem_right_1[sese] += 1
                    pos_right_1[ttt] += 1
                    if not pos_sememeNum.get(ttt):
                        pos_sememeNum[ttt] = [sese]
                    else:
                        pos_sememeNum[ttt].append(sese)
                else:
                    ttt = pos_name_2[word2pos_2[www]]
                    sememeNum2Char_2[sese] += 1
                    sem_right_2[sese] += 1
                    pos_right_2[ttt] += 1
                    if not pos_sememeNum.get(ttt):
                        pos_sememeNum[ttt] = [sese]
                    else:
                        pos_sememeNum[ttt].append(sese)
            except Exception as e:
                pass
            right_times += 1
            # print('pos:', pos_score, 'Neg:', neg_score,)
        else:
            # print('Neg:', neg_score, 'pos:', pos_score,)
            try:
                try:
                    sese = len(char2Sememe[www])
                except Exception as e:
                    sese = 99
                if len(dataset) < 10:
                    ttt = pos_name_1[word2pos_1[www]]
                    sem_wrong_1[sese] += 1
                    pos_wrong_1[ttt] += 1
                    sememeNum2Char_1[sese] += 1
                    if not pos_sememeNum.get(ttt):
                        pos_sememeNum[ttt] = [sese]
                    else:
                        pos_sememeNum[ttt].append(sese)
                else:
                    ttt = pos_name_2[word2pos_2[www]]
                    sem_wrong_2[sese] += 1
                    pos_wrong_2[ttt] += 1
                    sememeNum2Char_2[sese] += 1
                    if not pos_sememeNum.get(ttt):
                        pos_sememeNum[ttt] = [sese]
                    else:
                        pos_sememeNum[ttt].append(sese)

            except Exception as e:
                pass
            wrong_times += 1
    print("三元组分类成功率为: {} / {} ({})".format(right_times, right_times + wrong_times, round(right_times / (wrong_times + right_times) * 100, 1)))
    return round(right_times / (wrong_times + right_times) * 100, 2)

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


# patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
def drawPlot(x, y1, y2, name, en, patterns = ('/','\\')):
    labels = np.arange(len(x))
    men_means = np.array(y1)
    women_means = np.array(y2)
    width = 0.3  # the width of the bars: can also be len(x) sequence
    alpha = 0.6
    # plt.style.use("bmh")

    # print(labels, women_means, men_means)
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # x轴刻度标签位置
    ax.bar(labels - width/2, men_means, width, label='Frequency', alpha=alpha, hatch=patterns[0])
    ax.bar(labels + width/2, women_means, width, label='Accuracy', alpha=alpha, hatch=patterns[1])
    if max(men_means) > 0.2:
        gap = round(max(men_means) * 0.2, 2)
    else:
        gap = round(max(men_means) * 0.3, 2)
    ax.plot(labels, men_means + gap, linestyle='-.', color='b', marker='s', alpha=alpha, label="Frequency (+{})".format(gap))  # dash-dot line style, magenta, |
    ax.plot(labels, women_means + gap, linestyle=':', color='r', marker='v', alpha=alpha, label="Accuracy (+{})".format(gap)) # dotted line style, green
    if name.find('POS') != -1:
        ax.set_ylim(0., 0.55)
    else:
        ax.set_ylim(0., 0.18)


    # ax.bar(en, men_means, width, label='Frequency', hatch=patterns[0])
    # ax.bar(en, women_means, width, bottom=men_means, hatch=patterns[1],
    #        label='Accuracy')
    plt.xticks(x, labels=en)
    # ax.set_ylabel('Scores')
    ax.set_title('Frequency & Acc - %s'%name, fontsize=16, color='black')
    ax.legend(fontsize=14)
    for x in ax.get_xticklabels():
        x.set_rotation(45)  # 这里是调节横坐标的倾斜度，rotation是度数
    picname = ''
    if name.find('meme') != -1:
        ax.set_xlabel('Sememe Numbers')
        picname +='sememe-freq-'
    else:
        ax.set_xlabel('Part-Of-Speech')
        picname +='pos-freq-'
    if name.find('char') != -1:
        picname += 'charkg'
    else:
        picname += 'handian'

    if platform().lower().find('linux') != -1:
        plt.savefig('./{}.png'.format(picname))
    else:
        plt.savefig('/Users/username/Documents/Papers/Antonym/figure/{}.png'.format(picname))
        plt.show()


def checkTD():
    _, testList = openTrain('./data/test.txt')
    _, validList = openTrain('./data/valid.txt')
    _, tripleDisc = openTrain('./data/tripleDisc.txt')
    right = 0
    miss = 0
    for tri in testList:
        h,t,r = tri[0],tri[1],tri[2]
        if tri in tripleDisc or tuple([t,h,r]) in tripleDisc:
            right += 1
        else:
            miss += 1
    print('Triple Discovery的准确率:{} / {}, {}\n'.format(right, right+miss, round(right/(right+miss),4)))

def checkCharSememe():
    allchar = []
    for x in [tripleListTrain, tripleListTest, tripleListValid]:
        for l in x:
            a,b,r = l
            allchar.append(a)
            allchar.append(b)
    semNum2char = [0] * 100
    for c in allchar:
        try:
            sese = len(char2Sememe[c])
        except Exception as e:
            sese = 99
        semNum2char[sese] += 1
    precent = 0
    s_rr_1 = 0
    s_rr_2 = 0
    s_ww_1 = 0
    s_ww_2 = 0
    s_r_1 = 0
    s_r_2 = 0
    s_w_1 = 0
    s_w_2 = 0

    for i in range(99):
        if semNum2char[i] == 0:continue
        if i < 8 and i > 0:
            precent += semNum2char[i]/sum(semNum2char)
            print('{}/{}, R:{}, W:{}, R2:{}, W2:{}'.format(i,semNum2char[i]/sum(semNum2char), sem_right_1[i], sem_wrong_1[i], sem_right_2[i], sem_wrong_2[i]))
            s_r_1 += sem_right_1[i]
            s_w_1 += sem_wrong_1[i]
            s_r_2 += sem_right_2[i]
            s_w_2 += sem_wrong_2[i]
        else:
            s_rr_1 += sem_right_1[i]
            s_ww_1 += sem_wrong_1[i]
            s_rr_2 += sem_right_2[i]
            s_ww_2 += sem_wrong_2[i]
        # print('{}, {}/{}, {}'.format(i, semNum2char[i], sum(semNum2char), round(semNum2char[i]/sum(semNum2char), 4)))
    print(s_r_1, s_w_1, s_r_2, s_w_2, s_rr_1, s_ww_1, s_rr_2, s_ww_2)
    print('1-7 Acc: {} for CharKG, {} for HanDian, Other: {}, {}'.format(round(s_r_1 / (s_r_1 + s_w_1), 4),
                                                                         round(s_r_2 / (s_r_2 + s_w_2), 4),
                                                                         round(s_rr_1 / (s_rr_1 + s_ww_1), 4),
                                                                         round(s_rr_2 / (s_rr_2 + s_ww_2), 4),
                                                                         ))
    print(round(precent, 4))

if __name__ == '__main__':
    try:
        arg = int(sys.argv[1])
    except Exception as e:
        arg = 1
    try:
        emb_file = sys.argv[2]
    except Exception as e:
        emb_file = 'entityVector.txt'

    try:
        is_valid = sys.argv[3]
    except Exception as e:
        is_valid = 0

    dirTrain = "./data/train.txt"
    _, tripleListTrain = openTrain(dirTrain)
    dirTest = "./data/test.txt"
    _, tripleListTest = openTrain(dirTest)
    dirValid = "./data/valid.txt"
    _, tripleListValid = openTrain(dirValid)
    allTriples = []
    for x in [tripleListTrain, tripleListTest, tripleListValid]:
        allTriples += [(a,c,b) for a,b,c in x]
    # char_syns, char_ants = tongji1N()
    dirEntity = "./data/entity2id.txt"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    # FinishTODO 要接入词性分析，smeeme数量分析模块了。
    if is_valid:
        ranks_head, ranks_tail = link_prediction(arg, emb_file, './data/valid.txt')
    else:
        ranks_head, ranks_tail = link_prediction(arg, emb_file, './data/valid.txt')
    # if arg:
    #     print(ranks_head)
    head_1, head_3, head_10, head_20 = process_rank(ranks_head)
    tail_1, tail_3, tail_10, tail_20 = process_rank(ranks_tail)
    print("head: top1:{}, top5:{}, top10:{}, MRR:{}, MR:{}/ {}".format(head_1, head_3, head_10, round(np.mean([1/x for x in ranks_head]), 4), round(np.mean(ranks_head)), len(ranks_head)))
    print("tail: top1:{}, top5:{}, top10:{}, MRR:{}, MR:{}/ {}".format(tail_1, tail_3, tail_10, round(np.mean([1/x for x in ranks_tail]), 4), round(np.mean(ranks_tail)), len(ranks_tail)))
    print("Total: Hit@1:{}, Hit@3:{}, Hit@10:{}".format(round((head_1 + tail_1)/len(ranks_tail)/2,4),
                                                        round((head_3 + tail_3) / len(ranks_tail)/2,4),
                                                          round( (head_10 + tail_10) / len(ranks_tail)/2,4),
                                                        ))
    print("head + tail rank: 【==={}===】, MRR:{}".format((round(np.mean(ranks_head)) + round(np.mean(ranks_tail))) // 2,
                                              (round(np.mean([1 / x for x in ranks_tail]), 4) + round( np.mean([1 / x for x in ranks_head]), 4)) / 2))

    ranks_head, ranks_tail = link_prediction(arg, emb_file, 'data/anti_from_handian_filter.txt')
    # if arg:
    #     print(ranks_head)
    head_1, head_3, head_10, head_20 = process_rank(ranks_head)
    tail_1, tail_3, tail_10, tail_20 = process_rank(ranks_tail)
    print("head: top1:{}, top5:{}, top10:{}, MRR:{}, MR:{}/ {}".format(head_1, head_3, head_10,
                                                                       round(np.mean([1 / x for x in ranks_head]), 4),
                                                                       np.mean(ranks_head), len(ranks_head)))
    print("tail: top1:{}, top5:{}, top10:{}, MRR:{}, MR:{}/ {}".format(tail_1, tail_3, tail_10,
                                                                       round(np.mean([1 / x for x in ranks_tail]), 4),
                                                                       np.mean(ranks_tail), len(ranks_tail)))
    print("Total: Hit@1:{}, Hit@3:{}, Hit@10:{}".format(round((head_1 + tail_1)/len(ranks_tail)/2,4),
                                                        round((head_3 + tail_3) / len(ranks_tail)/2,4),
                                                          round( (head_10 + tail_10) / len(ranks_tail)/2,4),
                                                        ))
    print("head + tail rank: 【==={}===】, MRR:{}".format((round(np.mean(ranks_head)) + round(np.mean(ranks_tail))) // 2,
                                              (round(np.mean([1 / x for x in ranks_tail]), 4) + round(np.mean([1 / x for x in ranks_head]), 4))/2))

    # exit()
    word2pos_1, pos_name_1, needPos_1, needPosFreq_1 = pos_all_word('./data/test.txt', 'charKG')
    # print(word2pos_1, pos_name_1)
    word2pos_2, pos_name_2, needPos_2, needPosFreq_2 = pos_all_word('./data//anti_from_handian_filter.txt', 'handian')
    pos_right_1 = {x:0 for x in pos_name_1.values()}
    pos_wrong_1 = {x:0 for x in pos_name_1.values()}
    pos_right_2 = {x:0 for x in pos_name_2.values()}
    pos_wrong_2 = {x:0 for x in pos_name_2.values()}
    char2Sememe = openEntitySememe()
    sememeNum2Char_1 = [0] * 100
    sememeNum2Char_2 = [0] * 100
    sem_right_1 = [0] * 100
    sem_wrong_1 = [0] * 100
    sem_right_2 = [0] * 100
    sem_wrong_2 = [0] * 100
    pos_sememeNum = {}
    h1 = triple_classification(arg, emb_file, True, './data/test.txt')
    t1 = triple_classification(arg, emb_file, False, './data/test.txt')
    h2 = triple_classification(arg, emb_file, True, dataset='./data/anti_from_handian_filter.txt')
    # h2 = triple_classification(arg, emb, True, dataset='./data/anti_from_handian.txt')
    t2 = triple_classification(arg, emb_file, False, './data/anti_from_handian_filter.txt')
    # t2 = triple_classification(arg, emb, False, './data/anti_from_handian.txt')
    print("charKG, 54: 【---{}---】, HanDian 202:【---{}---】".format( round((h1+t1) / 2, 2), round((h2 + t2)/2,2)))
    try:
        os.system('cp {} vecs/entityVector_{}_{}.txt'.format(emb_file, round(np.mean(ranks_head)), round(np.mean(ranks_tail))))
    except Exception as e:
        pass
    pos_right_1_sorted = sorted(pos_right_1.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_right_2_sorted = sorted(pos_right_2.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_wrong_1_sorted = sorted(pos_wrong_1.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_wrong_2_sorted = sorted(pos_wrong_2.items(), key=lambda x: x[1], reverse=True)[:10]
    for x in ['动词', '名词', '形容词']:
        print(x, '%s / %s =  %s'%(pos_right_1[x], pos_right_1[x] + pos_wrong_1[x] ,pos_right_1[x] / (pos_right_1[x] + pos_wrong_1[x])))
        print(x, '%s / %s =  %s'%(pos_right_2[x], pos_right_2[x] + pos_wrong_2[x] ,pos_right_2[x] / (pos_right_2[x] + pos_wrong_2[x])))
    #
    pos_1 = {}
    pos_2 = {}
    for x in needPos_1:
        pos = x
        freq = pos_right_1[x]
        try:
            pos_1[pos] = freq / (freq + pos_wrong_1[pos])
        except Exception as e:
            pos_1[pos] = 0
    for x in needPos_2:
        pos = x
        freq = pos_right_2[x]
        try:
            pos_2[pos] = freq / (freq + pos_wrong_2[pos])
        except Exception as e:
            pos_2[pos]=0
    freq1 = list(pos_1.values())
    freq2 = list(pos_2.values())
    poszh2en_1 = ['verb', 'noun', 'adjective', 'person', 'place', 'state', 'pronoun', 'adverb', 'noun morpheme', 'adjective morpheme']
    poszh2en_2 = ['adjective', 'verb', 'noun', 'adverb', 'state word', 'locative', 'noun morpheme', 'person', 'place name', 'pronoun']
    drawPlot(needPos_1, needPosFreq_1, [needPosFreq_1[i] * freq1[i] for i in range(len(freq1))] , 'POS in charKG', poszh2en_1)
    drawPlot(needPos_2, needPosFreq_2, [needPosFreq_2[i] * freq1[i] for i in range(len(freq2))], 'POS in HanDian', poszh2en_2)

    freq3 = [sememeNum2Char_1[i]/np.sum(sememeNum2Char_1[:99]) for i in range(40)]
    freq4 = [sememeNum2Char_2[i]/np.sum(sememeNum2Char_2[:99]) for i in range(40)]
    acc3 = []
    acc4 = []
    for i in range(40):
        if i == 0:
            continue
        if sememeNum2Char_1[i] > 0:
            acc3.append(sem_right_1[i] / sememeNum2Char_1[i] * freq3[i])
        if sememeNum2Char_2[i] > 0:
            acc4.append(sem_right_2[i] / sememeNum2Char_2[i] * freq4[i])
    sememeNums1 = [x for x in range(len(freq3)) if freq3[x] != 0 and x > 0]
    sememeNums2 = [x for x in range(len(freq4)) if freq4[x] != 0 and x >0]
    freq3 = [x for x in freq3[1:] if x > 0]
    freq4 = [x for x in freq4[1:] if x > 0]
    drawPlot(sememeNums1, freq3,  acc3, 'Sememe Num in charKG', sememeNums1)
    drawPlot(sememeNums2, freq4,  acc4, 'Sememe Num in HanDian', sememeNums2)
    checkCharSememe()