import random
from copy import deepcopy

from testOpenHowNet import *
from tqdm import tqdm
syn = []
ant = []
chars_in_ant = set()
chars_in_syn = set()
chars_all = set()


# This is the pre-processed content we have, removing parts with length greater than 1 in the network synonyms and antonyms.
with open('./data/close_semantic.txt', 'r') as f:
    for l in f.readlines():
        a, b = l.strip().split()
        if (a, b) in syn or (b, a) in syn:
            continue
        syn.append((a,b))
        chars_in_syn.add(a)
        chars_in_syn.add(b)
        chars_all.add(a)
        chars_all.add(b)

with open('./data/anti_sem_char.txt', 'r') as f:
    for l in f.readlines():
        a, b = l.strip().split()
        if (a, b) in ant or (b, a) in ant:
            continue
        ant.append((a,b))
        chars_in_ant.add(a)
        chars_in_ant.add(b)
        chars_all.add(a)
        chars_all.add(b)

print("反义词有:{}组， 包含字符:{}个, 同义词有{}组，包含字符{}个，一共有字符:{}个".format(len(ant), len(chars_in_ant), len(syn), len(chars_in_syn), len(chars_all)))
triples = [(x, 'SYN', y) for x,y in syn] + [(x, 'ANT', y) for x, y in ant]

# To avoid entities in the test and validation sets that do not appear in the training set, replacement is necessary.

train_char_set = set()
train_list =[]
test_list =[]
valid_list =[]

# A portion of the data is taken in advance as the training set.
for idx in range(len(triples)):
    h, r, t = triples[idx]
    if idx % 10 < 8:
        train_char_set.add(h)
        train_char_set.add(t)
        train_list.append((h,t,r))

# Entities in the validation set are replaced.
for idx in tqdm(range(len(triples))):
    if idx % 10 == 8:
        h, r, t = triples[idx]
        if (h, t, r) in train_list:
            print(idx, "fuck?! Test", (h, t, r))
        # test_list.append((h, t, r))
        while True:
            if h in train_char_set and t in train_char_set:
                test_list.append((h,t,r))
                break
            else:
                seed = random.randint(0, len(train_list)-1)
                newdata = deepcopy(train_list[seed])
                while newdata[2] != r:
                    seed = random.randint(0, len(train_list) - 1)
                    newdata = deepcopy(train_list[seed])
                train_list[seed] = (h, t, r)
                h, t, r = newdata

# Entities in the test set are replaced.
for idx in tqdm(range(len(triples))):
    if idx % 10 > 8:
        h, r, t = triples[idx]
        # valid_list.append((h, t, r))
        while True:
            if h in train_char_set and t in train_char_set:
                valid_list.append((h,t,r))
                break
            else:
                seed = random.randint(0, len(train_list)-1)
                newdata = deepcopy(train_list[seed])
                while newdata[2] != r:
                    seed = random.randint(0, len(train_list) - 1)
                    newdata = deepcopy(train_list[seed])
                train_list[seed] = (h, t, r)
                h, t, r = newdata




with open('./data/train.txt', 'w') as train:
    for x in tqdm(train_list):
        h,t,r = x
        train.write('{}\t{}\t{}\n'.format(h, t, r))

with open('./data/test.txt', 'w') as test:
    for x in tqdm(test_list):
        h,t,r = x
        test.write('{}\t{}\t{}\n'.format(h, t, r))

with open('./data/valid.txt', 'w') as valid:
    for x in tqdm(valid_list):
        h,t,r = x
        valid.write('{}\t{}\t{}\n'.format(h, t, r))


with open('./data/entity2id.txt', 'w') as entity:
    for idx, e in enumerate(chars_all):
        entity.write('{}\t{}\n'.format(e, idx))

with open('./data/relation2id.txt', 'w') as entity:
    for idx, e in enumerate(['SYN', 'ANT']):
        entity.write('{}\t{}\n'.format(e, idx))

# Summary the following data and store them:

sense_in_ant = set()
sense_in_syn = set()
sense_all = set()
sememe_in_ant = set()
sememe_in_syn = set()
sememe_all = set()

sense_chars = {}
char_senses = {}
sememe_chars = {}
char_sememes = {}

for char in chars_in_syn:
    sense, sememes = getSenseSememe(char, False)
    if not char_senses.get(char):
        char_senses[char] = []
    if not char_sememes.get(char):
        char_sememes[char] = []
    for s in sense:
        sense_in_syn.add(s)
        sense_all.add(s)
        if not sense_chars.get(s):
            sense_chars[s] = []
        sense_chars[s].append(char)
        char_senses[char].append(s)
    for s in sememes:
        for x in s:
            sememe_in_syn.add(x)
            sememe_all.add(x)
            if not sememe_chars.get(x):
                sememe_chars[x] = []
            sememe_chars[x].append(char)
            char_sememes[char].append(x)

for char in chars_in_ant:
    sense, sememes = getSenseSememe(char, False)
    if not char_senses.get(char):
        char_senses[char] = []
    if not char_sememes.get(char):
        char_sememes[char] = []
    for s in sense:
        sense_in_ant.add(s)
        sense_all.add(s)
        if not sense_chars.get(s):
            sense_chars[s] = []
        sense_chars[s].append(char)
        char_senses[char].append(s)
    for s in sememes:
        for x in s:
            sememe_in_ant.add(x)
            sememe_all.add(x)
            if not sememe_chars.get(x):
                sememe_chars[x] = []
            sememe_chars[x].append(char)
            char_sememes[char].append(x)

with open('./data/entityWords.txt', 'w') as entityWords:
    for ent, sememe in char_sememes.items():
        sememe = set(sememe)
        entityWords.write('{}\t{}\t{}\n'.format(ent, len(sememe), " ".join([str(x) for x in sememe])))

with open('./data/entitySenses.txt', 'w') as entitySenses:
    for ent, sense in char_senses.items():
        sense = set(sense)
        entitySenses.write('{}\t{}\t{}\n'.format(ent, len(sense), " ".join([str(x) for x in sense])))

with open('./data/word2id.txt', 'w') as word2id:
    sememes = set()
    for ent, sememe in char_sememes.items():
        for x in sememe:
            sememes.add(x)
    for idx, sememe in enumerate(sememes):
        word2id.write('{}\t{}\n'.format(str(sememe), idx))


print("反义词sense:{}， 同义词sense:{}，一共有sense:{}个".format(len(sense_in_ant), len(sense_in_syn), len(sense_all)))
print("反义词sememe:{}， 同义词sememe:{}，一共有sememe:{}个".format(len(sememe_in_ant), len(sememe_in_syn), len(sememe_all)))




if __name__ == '__main__':
    pass
