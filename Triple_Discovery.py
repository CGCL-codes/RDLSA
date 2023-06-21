import os
from tqdm import tqdm
import Levenshtein
import numpy as np
from TFIDF import genCharSememeTFIDF, F1Score
from xgboot_sememe_relevance import *
THETA = 0.3
import json
sememeVec = 'SEMEME'
is_TFIDF = 'TFIDF'
sememeCosine = {}
recordLevCos = {}
xgtrain = open('./data/negXgcorpus.txt', 'w')
xgsememepair =  open('./data/negSememePairs.txt', 'w')
def openTrain(dir, sp="\t"):
    num = 0
    linelist = []
    with open(dir) as file:
        lines = file.readlines()
        for line in tqdm(lines):
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            a,b,c = triple
            if [b,a,c] in linelist:
                continue
            linelist.append(tuple(triple))
            num += 1
    return num, linelist


def tongji1N():
    print('Now TongJi All Triples')
    char_syns = {}
    char_ants = {}
    for tr in tqdm(tripleList):
        h, t, r = tr
        if r == 'SYN':
            if not char_syns.get(h):
                char_syns[h] = set()
            if not char_syns.get(t):
                char_syns[t] = set()
            char_syns[h].add(t)
            char_syns[t].add(h)
        if r == 'ANT':
            if not char_ants.get(h):
                char_ants[h] = set()
            if not char_ants.get(t):
                char_ants[t] = set()
            char_ants[h].add(t)
            char_ants[t].add(h)
    return char_syns, char_ants


def all_sememe_need(char2SememeFile='entityWords.txt'):
    needSememes = set()
    with open(char2SememeFile, 'r') as f:
        for line in tqdm(f.readlines()):
            data = line.split()
            sememes = data[2:]
            for x in sememes:
                needSememes.add(x[x.find('|') + 1:])
    return needSememes

# def checkSememeVec(vec='/home/usename/HanziNet/SenseEmbedding/glove.txt', char2SememeFile='entityWords.txt'):
def checkSememeVec(vec='data/sememe-vec.txt', char2SememeFile='entityWords.txt'):
    sememe_list = []
    embeddings = []
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
    return char2Sememe, sememe2index, sememe_list, embeddings


def distance(s1, s2, alpha=0.2, rel='ANT', f1=0.):
    # d_s 计算
    if (s1, s2, alpha) in sememeCosine:
        lev, cossim = recordLevCos["_".join([s1, s2, str(alpha), rel])]
        score = sememeCosine["_".join([s1, s2, str(alpha), rel])]
    else:
        sim = Levenshtein.jaro_winkler(s1, s2)
        e1 = np.array(embeddings[sememe2index[s1]])
        e2 = np.array(embeddings[sememe2index[s2]])
        vsim = e1.dot(e2.T) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        try:
            score = model.predict(pd.DataFrame([[f1, sim, vsim]], columns='0,1,2'.split(',')))[0]
        except Exception as _:
            score = sim + vsim + f1
        sememeCosine["_".join([s1, s2, str(alpha), rel])] = score
        sememeCosine["_".join([s2, s1, str(alpha), rel])] = score
        recordLevCos["_".join([s1, s2, str(alpha), rel])] = (sim, vsim)
        recordLevCos["_".join([s2, s1, str(alpha), rel])] = (sim, vsim)
        lev, cossim = sim, vsim
    xgtrain.write(' {} {} 1\n'.format(lev, cossim))
    return score


def calSimilarity(c1, c2, rel):
    global THETA
    sememe1 = char2Sememe[c1]
    sememe2 = char2Sememe[c2]
    if len(sememe2) == 0 or len(sememe1) == 0:
        return []
    if rel=='SYN':
        alpha = 0.3
    else:
        alpha = 0.1
    max1 = 0
    maxPair = None
    simSememePair = []
    scores = []
    # pair2score = {}
    for s1 in sememe1:
        for s2 in sememe2:
            if s1 == s2:
                continue
            if is_TFIDF == 'TFIDF':
                f1 = F1Score(CharSememeTFIDF[c1][s1], CharSememeTFIDF[c2][s2])
                xgtrain.write('{} '.format(f1))
                xgsememepair.write('{} {}\n'.format(s1, s2))
                score = distance(s1, s2, alpha, rel, f1)
            else:
                score = distance(s1, s2, alpha, rel)
            if score >= max1:
                max1 = score
                maxPair = [(s1, s2), (s2, s1)]
            if score < THETA:
                continue
            scores.append(score)
            simSememePair.append((s1, s2))
            simSememePair.append((s2, s1))
            # pair2score[(s1, s2, sim, vsim)] = score
    # sortedSim = sorted(pair2score.items(), key=lambda x: x[1], reverse=True)
    if len(simSememePair) == 0:
        if maxPair == None:
            return []
        simSememePair += maxPair
    return simSememePair


def checktTransitivity(triple1, triple2, commSememeList):
    if len(triple1) !=3 or len(triple2) != 3:
        return None
    x, y, r1 = triple1
    x, z, r2 = triple2
    r3 = None
    if r1 == r2:
        r3 = 'SYN'
    else:
        r3 = 'ANT'
    if (y, z, r3) in tripleList or (z, y, r3) in tripleList or z==y:
        return None
    commSememe1 = calSimilarity(x, y, r1)
    commSememe2 = calSimilarity(x, z, r2)
    commSememes = [x for x in commSememe1 if x in commSememe2]
    if len(commSememes) > 0:
        commSememeList.append([x, y, z, commSememes])
        return (y, z, r3)

def ExpandDataset():
    print('Now Expand Triple with our Algo~')
    add_syn = 0
    add_ant = 0
    char_syns, char_ants = tongji1N()
    newTriples = []
    commSememeList = []
    print('Expand Triple Step 1 ||||||||')
    for x in tqdm(char_syns):
        for y in char_syns[x]:
            # S + S = S
            for z in char_syns[x]:
                new = checktTransitivity((x,y,'SYN'), (x,z,'SYN'), commSememeList)
                if new:
                    if new in newTriples:
                        commSememeList.pop()
                        continue
                    newTriples.append(new)
                    add_syn += 1
            # S + A = A
            try:
                for z in char_ants[x]:
                    new = checktTransitivity((x, y, 'SYN'), (x, z, 'ANT'), commSememeList)
                    if new:
                        if new in newTriples:
                            commSememeList.pop()
                            continue
                        newTriples.append(new)
                        add_ant += 1
            except KeyError as e:
                # print('\r[%s] 没有反义词'%e, end='')
                pass
    print('Expand Triple Step 2 |||||||| ||||||||')
    for x in tqdm(char_ants):
        for y in char_ants[x]:
            # A + A = S
            for z in char_ants[x]:
                new = checktTransitivity((x, y, 'ANT'), (x, z, 'ANT'), commSememeList)
                if new:
                    if new in newTriples:
                        commSememeList.pop()
                        continue
                    newTriples.append(new)
                    add_syn += 1
            # A + S = A
            try:
                for z in char_syns[x]:
                    new = checktTransitivity((x, y, 'ANT'), (x, z, 'SYN'), commSememeList)
                    if new:
                        if new in newTriples:
                            commSememeList.pop()
                            continue
                        newTriples.append(new)
                        add_ant += 1

            except KeyError as e:
                # print('\r[%s] 没有同义词'%e, end='')
                pass

    with open('./tripleDiscovery/tripleDiscovery_{}_{}_{}.txt'.format(sememeVec, THETA, is_TFIDF), 'w') as f:
        for idx, t in enumerate(newTriples):
            print(t, commSememeList[idx])
            f.write('{}\t{}\t{}\n'.format(t[0], t[1], t[2]))
    print("新增的同义词对有:{}, "
          "新增反义词对有:{}, "
          "原有三元组:{}个,"
          "当前新增三元组{}个，"
          "共同Sememe：{}个".format(add_syn,
                                add_ant,
                                len(tripleList),
                                len(newTriples),
                                len(commSememeList)))

def check(a, b):
    print('当前可以检查在{}下，THETA设置为{}在TFIDF下的差异了'.format(a, b))
    tf = []
    with open('./tripleDiscovery/tripleDiscovery_{}_{}_{}.txt'.format(a,b,'TFIDF'), 'r') as f:
        for l in f.readlines():
            tf.append(l.strip())
    notf = []
    with open('./tripleDiscovery/tripleDiscovery_{}_{}_{}.txt'.format(a, b, 'noTFIDF'), 'r') as f:
        for l in f.readlines():
            notf.append(l.strip())
    with open('./tripleDiscovery/ReduceByTFIDF_tripleDiscovery_{}_{}.txt'.format(a,b), 'w') as f:
        for x in notf:
            if x not in tf:
                f.write(x+'\n')


def calNegTriples():
    for a,b,c in tqdm(tripleList):
        calSimilarity(a,b,c)


if __name__ == '__main__':

    is_used_to_discover_triples = False
    if is_used_to_discover_triples:
        tripleNum, tripleList = openTrain('./data/train.txt')
    else:
        dirTrain = "./data/negTrain.txt"
        # if wanna generate the full train data for xgboost,
        # please use 'train.txt' to generate item[0,1,2] for [gs, ls, cs] (next, add target 1 for item[3]),  the output name, please use 'xgcorpus.txt'
        # and 'negTrain.txt' (add target 1 for item[3]) and concat them, the output name, please use 'negXgcorpus.txt'
        tripleNum, tripleList0 = openTrain(dirTrain)
        tripleList = [x for x in tripleList0 ]
    CharSememeTFIDF = genCharSememeTFIDF()

    for i in ['/home/usename/HanziNet/SenseEmbedding/glove.txt']:
    # for i in ['data/sememe-vec.txt', '/home/usename/HanziNet/SenseEmbedding/glove.txt']:
        if i.find('sememe') != -1:
            sememeVec = 'SEMEME'
        else:
            sememeVec = 'GloVe'
        for theta in [0.3]:
        # for theta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for tfidf in ['TFIDF']:
            # for tfidf in ['TFIDF', 'noTFIDF']:
                THETA = theta
                is_TFIDF = tfidf
                print('当前使用的Vec:{}, theta值是:{}'.format(i, theta))
                if './tripleDiscovery/tripleDiscovery_{}_{}_{}.txt'.format(sememeVec, THETA, is_TFIDF) in os.listdir(
                        'tripleDiscovery'):
                    continue
                char2Sememe, sememe2index, sememe_list, embeddings = checkSememeVec(i)
                # choose one function to realize different work.
                if is_used_to_discover_triples:
                    # relation inference
                    ExpandDataset()
                else:
                    # generate dataset for xgboost classifer
                    calNegTriples()
    check(sememeVec, theta)
    with open('./data/sememeSim.json', 'w') as f:
        f.write(json.dumps(sememeCosine, indent=4))
    xgtrain.close()
    xgsememepair.close()