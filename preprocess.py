#!/usr/bin/env python

"""NER Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import gzip
import collections
import struct

# Your preprocessing, features construction, and word2vec code.

def transform_word(word):
    lw = word.lower()
    #can skip this -piazza id 56
    #lw = re.sub('\d+', 'NUMBER', lw)
    return lw


bio2id = {}

def generate_bio2id(filename):
    with open(filename, 'r') as f:
        for line in f:
            k, v = line.split()
            if k.startswith("B"):
                continue
            bio2id[k] = int(v)
    bio2id['<t>'] = max(bio2id.values())+1
    bio2id['</t>'] = max(bio2id.values())+1
    return bio2id


word2idx = {}

def get_embeddings():
    embeddings = {}
    with gzip.open('data/glove.6B.50d.txt.gz', 'rb') as f:
        for line in f:
            cemb = line.split()
            embeddings[cemb[0]] = np.array(cemb[1:], np.float64)
    return embeddings

def get_vocab(filelist, embeddings):
    idx_to_embedding = []
    idx_to_embedding.append(np.random.randn(50))
    idx_to_embedding.append(np.random.randn(50))
    idx_to_embedding.append(np.random.randn(50))
    #just for illustration we add these
    word2idx["<unk>"] = 1
    word2idx["<s>"] = 2
    word2idx["</s>"] = 3
    idx=4
    missing = collections.Counter()
    for filename in filelist:
        with open(filename, 'r') as f:
            for line in f:
                #get current word.
                #Right now, we use only lower case!
                cword = line.split()
                if cword:
                    cword = cword[2]
                    cword = transform_word(cword)
                    if cword not in word2idx:
                        if cword in embeddings:
                            word2idx[cword] = idx
                            idx_to_embedding.append(embeddings[cword])
                            idx += 1
                        else:
                            missing[cword] += 1
    print len(word2idx), "words have been counted"
    print len(missing), "unknown words"
    print missing.most_common(100)
    print len(idx_to_embedding)
    return np.array(idx_to_embedding, dtype=np.float32)


ngrams = []
cities = collections.defaultdict(float)
cities2 = collections.defaultdict(float)
countries = collections.defaultdict(float)
names = collections.defaultdict(float)
firstmale = collections.defaultdict(float)
firstfemale = collections.defaultdict(float)
last = collections.defaultdict(float)


def read_csv(filename, out, delim, col, ncol=None, fn=None, encoding='utf8'):
    with codecs.open(filename, 'r', encoding) as f:
        for line in f:
            line = re.sub(r'".*?"', '""', line)
            cline = filter(None, line.split(delim))
            if len(cline) > col:
                word = cline[col]
                word = re.sub(r'[^a-z]+', '', word.lower())
                if word:
                    num = fn(cline[ncol]) if ncol else 1
                    out[word] += num
    print filename, len(out), "entries"


def read_json(filename, out, encoding='utf8'):
    with codecs.open(filename, 'r', encoding) as f:
        for line in f:
            cline = filter(None, re.split(r'[:",\]\[{}]+', line))
            for word in cline:
                word = re.sub(r'[^a-z]+', '', word.lower())
                if word:
                    out[word] += 1
    print filename, len(out), "entries"


def count_ngrams(filename):
    global ngrams
    ngrams = [collections.Counter() for i in xrange(5)]
    with open(filename, 'r') as f:
        for line in f:
            cline = line.split()
            if cline:
                word = cline[2]
                word = word.lower()
                word = re.sub(r'[^a-z]+', '', word)
                for n in xrange(2, len(ngrams)):
                    for i in xrange(len(word) - n + 1):
                        ngrams[n][word[i:i+n]] += 1
    for i in xrange(len(ngrams)):
        ngrams[i] = [w for w, c in ngrams[i].most_common(25)]
        print ngrams[i]


def get_features(word, window):
    res = []
    res.append(word2idx.get(transform_word(word), word2idx["<unk>"]))
    for oth in window:
        res.append(word2idx.get(transform_word(oth), word2idx["<unk>"]))

    res.append(len(word))
    for i in xrange(ord('0'), ord('9')+1):
        res.append(word.count(chr(i)))
    for c in ',.():;-\'\"</':
        res.append(word.count(c))
    for i in xrange(ord('a'), ord('z')+1):
        res.append(word.count(chr(i)))
    for i in xrange(ord('A'), ord('Z')+1):
        res.append(word.count(chr(i)))
    for i in xrange(5):
        res.append(word[i].isupper() if len(word) > i else 0)
    for i in xrange(5):
        res.append(word[i].islower() if len(word) > i else 0)
    for oth in window:
        res.append(len(oth))
        for i in xrange(2):
            res.append(oth[i].isupper() if len(oth) > i else 0)
        for i in xrange(2):
            res.append(oth[i].islower() if len(oth) > i else 0)

    word = re.sub(r'[^a-z]+', '', word.lower())

    res.append(word in cities)
    res.append(word in cities2)
    res.append(word in countries)
    res.append(word in names)
    res.append(word in firstmale)
    res.append(word in firstfemale)
    res.append(word in last)

    res.append(cities.get(word, 0))
    res.append(cities2.get(word, 0))
    res.append(countries.get(word, 0))
    res.append(names.get(word, 0))
    res.append(firstmale.get(word, 0))
    res.append(firstfemale.get(word, 0))
    res.append(last.get(word, 0))
    # for grams in ngrams:
    #     for w in grams:
    #         res.append(word.count(w))
    return res

# window on each side
WINDOW = 3

def get_all_features(words):
    features = []
    padded_words = ["<s>"] * WINDOW + words + ["</s>"] * WINDOW
    for i in xrange(len(words)):
        prev = padded_words[i: i + WINDOW]
        nxt = padded_words[i + WINDOW + 1: i + WINDOW * 2 + 1]
        features.append(get_features(words[i], prev + nxt))
    npfeatures = np.array(features, dtype=np.float32)
    return npfeatures

def convert_data(filename):
    #start tags
    words = ["<s>"]
    targets = [bio2id['<t>']]
    with open(filename, 'r') as f:
        for line in f:
            cline = line.split()
            if cline:
                word = cline[2]
                ctag = cline[3]
                words.append(word)
                if ctag.startswith("B"):
                    ctag = "I" + ctag[1:]
                targets.append(bio2id[ctag])
            else:
                words.append("</s>")
                words.append("<s>")

                targets.append(bio2id["</t>"])
                targets.append(bio2id["<t>"])
        #end last sentence
        words.append("</s>")
        targets.append(bio2id["</t>"])

    nptargets = np.array(targets[:-2], dtype=np.int32)
    return get_all_features(words[:-2]), nptargets


def convert_test(filename):
    #start tags
    words = ["<s>"]
    with open(filename, 'r') as f:
        for line in f:
            cline = line.split()
            if cline:
                word = cline[2]
                words.append(word)
            else:
                words.append("</s>")
                words.append("<s>")
        #end last sentence
        words.append("</s>")
    return get_all_features(words[:-2])


FILE_PATHS = {"CONLL": ("data/train.num.txt",
                        "data/dev.num.txt",
                        "data/test.num.txt",
                        "data/tags.txt")}
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset

    read_csv("data/countries.txt", countries, ":", 1)
    read_json("data/cities.txt", cities, encoding="utf_16_le")
    read_csv("data/cities2.txt", cities2, ",", 1, encoding="latin1")
    div = lambda a: int(a) * 0.001
    read_csv("data/names.csv", names, ",", 1, 2, div, encoding="latin1")
    read_csv("data/firstmale.txt", firstmale, " ", 0, 1, float)
    read_csv("data/firstfemale.txt", firstfemale, " ", 0, 1, float)
    read_csv("data/last.txt", last, " ", 0, 1, float)

    train, valid, test, tag_dict = FILE_PATHS[dataset]
    generate_bio2id(tag_dict)
    embeddings = get_embeddings()
    idx2embedding = get_vocab([train,valid,test], embeddings)
    #for every sentence, add start and end tag! <s> -> <t>, </s> -> </t>

    count_ngrams(train)
    train_input, train_target = convert_data(train)
    valid_input, valid_target = convert_data(valid)
    test_input = convert_test(test)

    C = len(bio2id)
    V = len(word2idx)

    print "nwords", V
    print "nclasses", C
    print "nfeatures", train_input.shape[1]

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_target

        f['valid_input'] = valid_input
        f['valid_output'] = valid_target

        f['test_input'] = test_input

        f['embeddings'] = idx2embedding
        f['nwords'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
