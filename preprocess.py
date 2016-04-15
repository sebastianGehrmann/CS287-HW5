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

# Your preprocessing, features construction, and word2vec code.

def transform_word(word):
    lw = word.lower()
    #can skip this -piazza id 56
    #lw = re.sub('\d+', 'NUMBER', lw)
    return lw

def generate_bio2id(filename):
    bio2id = {}
    with open(filename, 'r') as f:
        for line in f:
            k,v =line.split()
            bio2id[k] = int(v)
    bio2id['<t>'] = max(bio2id.values())+1
    bio2id['</t>'] = max(bio2id.values())+1
    return bio2id

def get_embeddings():
    embeddings = {}
    with gzip.open('data/glove.6B.50d.txt.gz', 'rb') as f:
        for line in f:
            cemb = line.split()
            embeddings[cemb[0]] = np.array(cemb[1:], np.float64)
    return embeddings

def get_vocab(filelist, embeddings):
    word2idx = {}
    idx_to_embedding = []
    idx_to_embedding.append(np.random.randn(50))
    idx_to_embedding.append(np.random.randn(50))
    idx_to_embedding.append(np.random.randn(50))
    #just for illustration we add these
    word2idx["<unk>"] = 1
    word2idx["<s>"] = 2
    word2idx["</s>"] = 3
    idx=4
    for filename in filelist:
        with open(filename, 'r') as f:
            for line in f:
                #get current word.
                #Right now, we use only lower case!
                cword = line.split()
                if cword:
                    cword = cword[2]
                    cword = transform_word(cword)
                    if cword not in word2idx and cword in embeddings:
                        word2idx[cword] = idx
                        idx_to_embedding.append(embeddings[cword])
                        idx += 1
    print(len(word2idx), "words have been counted")
    print(len(idx_to_embedding))
    return word2idx, np.array(idx_to_embedding, dtype=np.float32)

ngrams = []

def get_features(word):
    res = []
    res.append(len(word))
    for i in xrange(ord('0'), ord('9')+1):
        res.append(word.count(chr(i)))
    for i in xrange(ord('a'), ord('z')+1):
        res.append(word.count(chr(i)))
    for i in xrange(ord('A'), ord('Z')+1):
        res.append(word.count(chr(i)))
    for c in ',.():;-\'\"</':
        res.append(word.count(c))
    for i in xrange(5):
        res.append(word[i].isupper() if len(word) > i else 0)
    for i in xrange(5):
        res.append(word[i].islower() if len(word) > i else 0)
    # word = re.sub(r'[^a-z]+', '', word.lower())
    # for grams in ngrams:
    #     for w in grams:
    #         res.append(word.count(w))
    return res


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


def convert_data(filename, word2idx, bio2id):
    #start tags
    inputs = [word2idx['<s>']]
    features = [get_features('<s>')]
    targets = [bio2id['<t>']]
    with open(filename, 'r') as f:
        for line in f:
            cline = line.split()
            if cline:
                word = cline[2]
                #transformation still very basic!
                cword = transform_word(word)
                ctag = cline[3]
                if cword in word2idx:
                    inputs.append(word2idx[cword])
                else:
                    inputs.append(word2idx["<unk>"])
                features.append(get_features(word))
                targets.append(bio2id[ctag])
            else:
                inputs.append(word2idx["</s>"])
                inputs.append(word2idx["<s>"])

                features.append(get_features("</s>"))
                features.append(get_features("<s>"))

                targets.append(bio2id["</t>"])
                targets.append(bio2id["<t>"])
        #end last sentence
        inputs.append(word2idx["</s>"])
        features.append(get_features("</s>"))
        targets.append(bio2id["</t>"])
    npinput = np.array(inputs[:-2], dtype=np.int32)
    npinput.shape = (npinput.shape[0], 1)
    npfeatures = np.array(features[:-2], dtype=np.int32)
    nptargets = np.array(targets[:-2], dtype=np.int32)
    return np.hstack((npinput, npfeatures)), nptargets


def convert_test(filename, word2idx):
    #start tags
    inputs = [word2idx['<s>']]
    features = [get_features('<s>')]
    with open(filename, 'r') as f:
        for line in f:
            cline = line.split()
            if cline:
                word = cline[2]
                #transformation still very basic!
                cword = transform_word(word)
                if cword in word2idx:
                    inputs.append(word2idx[cword])
                else:
                    inputs.append(word2idx["<unk>"])
                features.append(get_features(word))
            else:
                inputs.append(word2idx["</s>"])
                inputs.append(word2idx["<s>"])
                features.append(get_features("</s>"))
                features.append(get_features("<s>"))
        #end last sentence
        inputs.append(word2idx["</s>"])
        features.append(get_features("</s>"))
    npinput = np.array(inputs[:-2], dtype=np.int32)
    npinput.shape = (npinput.shape[0], 1)
    npfeatures = np.array(features[:-2], dtype=np.int32)
    return np.hstack((npinput, npfeatures))


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
    train, valid, test, tag_dict = FILE_PATHS[dataset]

    bio2id = generate_bio2id(tag_dict)
    embeddings = get_embeddings()
    word2idx, idx2embedding = get_vocab([train,valid,test], embeddings)
    #for every sentence, add start and end tag! <s> -> <t>, </s> -> </t>

    count_ngrams(train)
    train_input, train_target = convert_data(train, word2idx, bio2id)
    valid_input, valid_target = convert_data(valid, word2idx, bio2id)
    test_input = convert_test(test, word2idx)

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
