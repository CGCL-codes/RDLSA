import os
import random
import re
import time
import urllib.request
from functools import reduce
from tqdm import tqdm
from bs4 import BeautifulSoup
import urllib.parse
import ssl
from Evaluation import read_file

handian_url = 'https://hanyu.baidu.com/s?wd=char&from=zici'


def post_baidu(url):
    # print(url)
    try:
        timeout = 5
        request = urllib.request.Request(url)
        # 伪装HTTP请求
        request.add_header('User-agent',
                           'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')
        request.add_header('connection', 'keep-alive')
        request.add_header('referer', url)
        myssl = ssl.create_default_context()
        myssl.check_hostname = False
        myssl.verify_mode = ssl.CERT_NONE
        # request.add_header('Accept-Encoding', 'gzip')  # gzip可提高传输速率，但占用计算资源
        response = urllib.request.urlopen(request, timeout=timeout, context=myssl)
        html = response.read()
        # if(response.headers.get('content-encoding', None) == 'gzip'):
        #    html = gzip.GzipFile(fileobj=StringIO.StringIO(html)).read()
        response.close()
        return html
    except Exception as e:
        print('URL Request Error:', e)
        return None


def get_soup(char):
    word_encode = urllib.parse.quote(char)
    url = handian_url.replace('char', word_encode)
    # print(url)
    html = post_baidu(url)
    if html == None:
        return None
    # else:
    #     if not debug_model:
    #         return 0

    # Step2 解析文件
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def getAntonyms(char):
    time.sleep(0.1)
    soup = get_soup(char)
    ss = soup.find_all('div', attrs={'id':'antonym'})
    try:
        ss = ss[0]
    except Exception as e:
        return []
    ants = []
    for x in ss.findAll('a'):
        a = x.text.strip()
        if len(a) != 1:
            print(char, a)
        else:
            ants.append(a)
    return ants

def testAntonymsDetection(dataset='./data/test.txt'):
    test_triples = read_file(dataset, True)
    head_ = 0
    tail_ = 0
    times = 0
    for tri in tqdm(test_triples):
        if tri[1] == 'SYN':
            continue
        times += 2
        h, _, t  = tri
        hAnts = getAntonyms(h)
        if t in hAnts:
            head_ += 1
        tAnts = getAntonyms(t)
        if h in tAnts:
            tail_ += 1
    print("三元组分类成功率为: head/{}  tail/{} all/{}, 准确率为:{}".format(head_, tail_, times, round((head_ + tail_ )/times * 100, 2)))



if __name__ == '__main__':
    testAntonymsDetection('./data/test.txt')
    testAntonymsDetection('./data/anti_from_handian.txt')