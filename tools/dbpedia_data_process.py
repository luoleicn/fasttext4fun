#!/usr/bin/python
#coding=utf8

import sys, os, re, random

script, train_dat, test_dat, featurefile, trainfile, testfile = sys.argv

cop = re.compile("[^a-z^A-Z^0-9]")


def newsgroup_reader(fn):
    f = open(fn, encoding="utf-8")
    try:
        text = ''.join(f.readlines())
        text = strip_newsgroup_footer(text)
        text = strip_newsgroup_header(text)
        text = strip_newsgroup_quoting(text)
        text = text.split("\n")
    except UnicodeDecodeError:
        text = []
    
    header = True
    tokens = []
    for line in text:
        line = line.lower()
        vec = line.split()
        vec = [cop.sub('', v) for v in vec if len(cop.sub('', v)) > 2]
        vec = [v for v in vec if v not in stopwords]
        tokens.extend(vec)
    f.close()
    return tokens

def get_label2id(fn):
    ret = {}
    n = 0
    for line in open(fn, encoding="utf-8"):
        vec = line.split(",")
        l = vec[0].strip()
        if l not in ret:
            ret[l] = n
            n += 1
    return ret

def get_labeltree(num_leaves):
    sons = [i for i in range(num_leaves)]
    link = []
    label_count = num_leaves

    while len(sons) > 1:
        fathers = []
        for i in range(0, len(sons), 2):
            left = sons[i]
            right = -1 if len(sons) <= i + 1 else sons[i+1]
            link.append((label_count, left, right))
            fathers.append(label_count)
            label_count += 1
        sons = fathers
    return list(reversed(link)), label_count

def get_featuremap(train_dat_file):

    tokens = set([])
    for line in open(train_dat_file, encoding="utf-8"):
        vec = line[:-1].split(",")
        for i in range(1, len(vec)):
            splits = vec[i].split()
            for s in splits:
                if len(s.strip()) > 2:
                    tokens.add(s.strip())
    featuremap = {}
    num_feature = 0
    for k in tokens:
        featuremap[k] = num_feature
        num_feature += 1

    return featuremap

def create_featurefile(train_dat_file, label2id):
    featuremap = get_featuremap(train_dat_file)
    labeltree, num_label = get_labeltree(len(label2id))

    f = open(featurefile, "w", encoding="utf-8")
    f.write(str(len(featuremap)) + "\n")
    for k, v in featuremap.items():
        f.write("{} {}\n".format(v, k))
    f.write(str(num_label) + "\n")
    for k, v in label2id.items():
        f.write("{} {}\n".format(v, k))
    for i in range(len(label2id), num_label):
        f.write(str(i) + " TreeNode\n")
    f.write(str(len(labeltree)) + "\n")
    for (father, left, right) in labeltree:
        f.write("{} {} {}\n".format(father, left, right))
    f.close()
    return featuremap, label2id

def get_labelid_from_filename(fn, label2id):
    for k, v in label2id.items():
        if k in fn:
            return v
    return -1

def record(writer, fn, label2id, featuremap):

    for line in open(fn, encoding="utf-8"):
        vec = line[:-1].split(",")
        labelid = label2id[vec[0].strip()]
        output = str(labelid) + " "
        for i in range(1, len(vec)):
            splits = vec[i].split()
            for s in splits:
                if s.strip() in featuremap:
                    output += str(featuremap[s.strip()]) + " "
        writer.write(output[:-1] + "\n")


def data_transformer():

    label2id   = get_label2id(train_dat)

    featuremap, label2id = create_featurefile(train_dat, label2id)
    trainwriter = open(trainfile, "w", encoding="utf-8")
    testwriter  = open(testfile, "w", encoding="utf-8")

    record(trainwriter, train_dat, label2id, featuremap)
    record(testwriter, test_dat, label2id, featuremap)

    trainwriter.close()
    testwriter.close()

data_transformer()

