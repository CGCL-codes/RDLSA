
import jieba.posseg as psg
import jieba

import matplotlib.pyplot as plt


def pos_(text):
    return psg.cut(text)

def pos_all_word(path, name, draw=False):
    pos_name = {'a': '形容词', 'ad': '副形词', 'an': '名形词', 'ag': '形容词性语素', 'b': '区别词', 'zg': '状态词', 'uv': '结构助词', 'c': '连词', 'dg': '副语素', 'eng': '英文', 'ng': '名词性语素', 'yg': '语气词', 'mq': '数量词', 'd': '副词', 'e': '叹词', 'nrt': '机构', 'f': '方位词', 'g': '语素', 'h': '前接成分', 'i': '成语', 'j': '简称', 'k': '后接成分', 'l': '习用语', 'm': '数词', 'Ng': '名语素', 'nrfg': '名词', 'n': '名词', 'nr': '人名', 'ns': '地名', 'nt': '团体', 'nz': '其他', 'o': '拟声词', 'p': '介词', 'q': '量词', 'r': '代词', 's': '处所词', 'tg': '时语素', 't': '时间词', 'u': '助词', 'vg': '动语素', 'v': '动词', 'vd': '副动词', 'vn': '名动词', 'w': '标点符号', 'x': '非语素字', 'y': '语气词', 'z': '状态词', 'un': '未知词'}

    word_list = read_wordList(path,'pair')
    poses = []
    word2pos = {}
    for i in word_list:
        x = pos_(i)
        for s in x:
            if s.word in word_list:
                poses.append(s)
            word2pos[s.word] = s.flag
    pos_freq = {}
    for i in poses:
        try:
            pos = pos_name[i.flag]
        except Exception as e:
            continue
        if not pos_freq.get(pos):
            pos_freq[pos] = 1
        else:
            pos_freq[pos] += 1
    cf = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)
    x=[x[0] for x in cf]
    y=[s[1] for s in cf]
    af = sum(y)
    for i in pos_freq.keys():
        pos_freq[i] = round(pos_freq[i]/af,3)
    cf = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)
    # print(cf)
    x=[x[0] for x in cf][:10]
    # print(x)
    y=[s[1] for s in cf][:10]
    # y = [round(x/af,3) for x in y]

    if draw:
        fig = plt.figure(figsize=(int(len(x)*0.7),4))

        plt.bar(x,y)
        # plt.bar(x,y,tick_label=['"%s"'%xx for xx in x])
        plt.title('Frequency - POS - %s'%name)
        # plt.xticks(())# 关闭x轴标签

        plt.xlabel('POS')

        plt.ylabel('Frequency')
        plt.savefig('Freq_pos-%s'%path[path.rfind('/')+1:path.rfind('.')])
        plt.show()
    return word2pos, pos_name, x, y


def read_wordList(vec_file, tag=''):
    # input:  the file of word2vectors
    # output: word dictionay, embedding matrix -- np ndarray
    f = open(vec_file,'r')
    cnt = 0
    word_list = []
    embeddings = []
    word_size = 0
    embed_dim = 0
    for line in f:
        if line.find(':')!=-1:
            continue
        data = line.split()
        if data[2] == 'SYN':
            continue
        if cnt == 0 and tag!='pair' and tag!='ana':
            word_size = int(data[0])
            embed_dim = int(data[1])
        else:
            try:
                if tag=='pair':
                    word_list.append(data[0])
                    word_list.append(data[1])
                elif tag=='ana':
                    word_list.append(data[0])
                    word_list.append(data[1])
                    word_list.append(data[2])
                    word_list.append(data[3])
                else:
                    word_list.append(data[0])
            except Exception as e:
                print(e, data[0])
        cnt += 1
    f.close()
    existDict = set()
    with open('data/user_dict.txt', 'r', encoding='utf8') as f:
        for i in f.readlines():
            existDict.add(i.strip())
    with open('data/user_dict.txt', 'w', encoding='utf8') as f:
        # print('generate Dict')
        for i in set(word_list):
            existDict.add(i + ' ' + str(1000000))
        for i in existDict:
            f.write(i+'\n')
    if tag=='pair' or tag=='ana':
        return list(set(word_list))
    return set(word_list)


if __name__ == '__main__':
    # char2Sememe, sememe2index, sememe_list, embeddings = checkSememeVec()
    #
    name = ['charKG', 'HanDian']
    for idx, p in enumerate(['./data/train.txt', './data/anti_from_handian_filter.txt']):
        pos_all_word(p, name[idx], True)
        print('=======')



