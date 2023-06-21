import OpenHowNet
import platform
import sys
from tqdm import tqdm
if str(platform.system()).lower().find('nux') != -1:
    BASE_DIR = '/home/usename/HanziNet/'
    HOME_DIR = '/home/usename'
else:
    BASE_DIR = '/Users/username/PycharmProjects/HanziNet/'
    HOME_DIR = '/Users/username/PycharmProjects'
sys.path.append(BASE_DIR)



hownet_dict = OpenHowNet.HowNetDict()

def getSenseSememe(char, show=True):
    """
    This function takes a Chinese character as input and retrieves its senses and sememes from the OpenHowNet knowledge base.
    If show parameter is set to True, it will print out the senses and their corresponding sememes to the console.
    The function returns the senses and sememes as a tuple if show parameter is set to False.
    """
    sense = hownet_dict.get_sense(char)
    sememes = []
    for s in sense:
        sememes.append(s.get_sememe_list())
    if not show:
        return sense, sememes
    for idx in range(len(sense)):
        print(sense[idx])
        print(sense[idx])
        print(' ---> ', end='')
        print(sememes[idx])
        print(len(sememes[idx]))


def openSememeVec(vec='data/sememe-vec.txt'):
    sememe_list = []
    embeddings = []
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
                tmpVec = [float(x) for x in data[-embed_dim:]]
                sememe_list.append(data[0])
                embeddings.append(tmpVec)
            cnt = cnt + 1
    sememe2index = {sememe_list[k]:k for k in range(len(sememe_list))}
    return sememe2index, embeddings



sememe2index, embeddings = openSememeVec()


def getAllWord_Sememe(word=''):
    # create empty dictionaries to store the results
    sense2Sememe = {}
    word2Sense = {}
    word2Sememe = {}

    # if a specific word is given, only process that word, otherwise process all words in the dictionary
    if word:
        all_word = [word]
    else:
        all_word = hownet_dict.get_zh_words()

    # iterate over all words
    for word in tqdm(all_word):
        # get all senses of the word
        sense = hownet_dict.get_sense(word)
        # map each word to its senses
        word2Sense[word] = sense
        # iterate over the senses of the word
        for s in sense:
            # if the sense is not already in the dictionary, add it with its sememes
            if not sense2Sememe.get(s):
                sense2Sememe[s] = s.get_sememe_list()
            # map each sense to its sememes
            sememe = [x for x in sense2Sememe[s] if x in sememe2index]
            # if the word is not already in the dictionary, add it with its sememes
            if not word2Sememe.get(word):
                word2Sememe[word] = []
                word2Sememe[word] = list(set(word2Sememe[word] + sememe))

    # create a dictionary that maps sememes to words
    sememe2Word = {}
    for k, v in word2Sememe.items():
        for vv in v:
            if not sememe2Word.get(vv):
                sememe2Word[vv] = [k]
            else:
                sememe2Word[vv].append(k)

    # write the results to file
    with open('data/word2Sense.txt', 'w') as ws:
        for k, v in word2Sense.items():
            if len(k) == 0:
                print('K:', k, ', ', v)
                continue
            ws.write("{}\t{}\t{}\n".format(k, len(v), " ".join([str(x) for x in v])))
    with open('data/word2Sememe.txt', 'w') as ws:
        for k, v in word2Sememe.items():
            if len(k) == 0:
                print('K:', k, ', ', v)
                continue
            ws.write("{}\t{}\t{}\n".format(k, len(v), " ".join([str(x) for x in v])))
    with open('data/sense2Sememe.txt', 'w') as ss:
        for k, v in sense2Sememe.items():
            if len(str(k)) == 0:
                print('K:', k, ', ', v)
                continue
            ss.write("{}\t{}\t{}\n".format(k, len(v), " ".join([str(x) for x in v])))
    with open('data/sememe2Word.txt', 'w') as sw:
        for k, v in sememe2Word.items():
            if len(str(k)) == 0:
                print('K:', k, ', ', v)
                continue
            sw.write("{}\t{}\t{}\n".format(k, len(v), " ".join([str(x) for x in v])))

if __name__ == '__main__':
    for x in '冷凉寒火热炎烫':
        getSenseSememe(x)
        print('=========')
