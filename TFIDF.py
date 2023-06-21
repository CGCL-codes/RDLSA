
import math
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def drawHist(data):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data, bins=120, facecolor="blue", edgecolor="black")
    # 显示横轴标签
    plt.xlabel("Sememes")
    # 显示纵轴标签
    plt.ylabel("Frequency")
    # 显示图标题
    plt.title("Sememe-Frequency")
    plt.show()

def checkSememeVec(char2SememeFile='./data/entityWords.txt'):
    with open(char2SememeFile, 'r') as f:
        char2Sememe = {}
        for line in tqdm(f.readlines()):
            try:
                data = line.split()
                char = data[0]
                length = int(data[1])
                sememes = data[2:]
                newsememes = []
                if length != len(sememes):
                    print(line)
                for x in sememes:
                    newsememes.append(x[x.find('|') + 1:])
                char2Sememe[char] = newsememes
            except Exception as e:
                continue
    return char2Sememe


def tf(sememe, count):
    """Calculate term frequency (tf) of a sememe in a count dictionary."""
    return count[sememe] / sum(count.values())

def n_containing(sememe, count_list):
    """Count the number of count dictionaries that contain a given sememe."""
    return sum(1 for count in count_list if sememe in count)

def idf(sememe, count_list):
    """Calculate inverse document frequency (idf) of a sememe in a list of count dictionaries."""
    return math.log(len(count_list) / (1 + n_containing(sememe, count_list)))

def tfidf(sememe, count, count_list):
    """Calculate tf-idf score of a sememe in a count dictionary, given a list of count dictionaries."""
    return tf(sememe, count) * idf(sememe, count_list)


def count_term(char, char2Sememe):
    """Count the occurrences of each sememe in the list of sememes associated with a given character."""
    count = Counter(char2Sememe[char])
    return count

def F1Score(a, b):
    """Calculate F1 score given precision a and recall b."""
    return 2 * a * b / (a + b)

def genCharSememeTFIDF(char2SememeFile='./data/entityWords.txt'):
    """Generate a dictionary of tf-idf scores for each sememe associated with each character in the input file."""
    char2Sememe = checkSememeVec(char2SememeFile)
    chars = list(char2Sememe.keys())
    countlist = []
    sememe2charNum = {}
    sememes = set()
    for x in char2Sememe.values():
        for y in x:
            sememes.add(y)
    for x in tqdm(sememes):
        sememe2charNum[x] = 0
    for char in chars:
        countlist.append(count_term(char, char2Sememe))
        for se in char2Sememe[char]:
            sememe2charNum[se] += 1
    # freqs = sorted(sememe2charNum.values())
    # drawHist(freqs)
    charSememeTFIDF = {}
    for i, count in tqdm(enumerate(countlist)):
        charSememeTFIDF[chars[i]] = {}
        # print("Top Sememes in Char {}".format(chars[i]))
        scores = {sememe: tfidf(sememe, count, countlist) for sememe in count}
        allScore = sum(scores.values())
        sorted_sememes = sorted(scores.items(), key = lambda x: x[1], reverse=True)
        for sememe, score in sorted_sememes:
            # print("\tSememe: {}, TF-IDF: {}".format(sememe, round(score/allScore, 5)))
            charSememeTFIDF[chars[i]][sememe] = round(score/allScore, 5)
    print(len(charSememeTFIDF))
    return charSememeTFIDF

if __name__ == "__main__":
    genCharSememeTFIDF()