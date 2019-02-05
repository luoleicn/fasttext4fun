#!/usr/bin/python
#coding=utf8

import sys, os, re, random

script, corpus, stopwordfile, featurefile, trainfile, valfile, testfile = sys.argv

cop = re.compile("[^a-z^A-Z^0-9]")
stopwords = set([])
for line in open(stopwordfile):
    stopwords.add(line[:-1].lower())

def strip_newsgroup_header(text):
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


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

def get_label2id():
    ret = {}
    ret["talk.politics.guns"] = 0
    ret["talk.politics.mideast"] = 1
    ret["talk.politics.misc"] = 2
    ret["talk.religion.misc"] = 3
    ret["rec.autos"] = 4
    ret["rec.motorcycles"] = 5
    ret["rec.sport.baseball"] = 6
    ret["rec.sport.hockey"] = 7
    ret["sci.crypt"] = 8
    ret["sci.electronics"] = 9
    ret["sci.med"] = 10
    ret["sci.space"] = 11
    ret["comp.sys.ibm.pc.hardware"] = 12
    ret["comp.sys.mac.hardware"] = 13
    ret["comp.os.ms-windows.misc"] = 14
    ret["comp.windows.x"] = 15
    ret["comp.graphics"] = 16
    ret["alt.atheism"] = 17
    ret["misc.forsale"] = 18
    ret["soc.religion.christian"] = 19
    return ret

def get_labeltree(num_leaves):
    """
    1:40
    2:38~39
    3:35~37
    5:30~34
    10:20~29
    20:0~19
    """
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

def get_featuremap(train_files):

    fid_count = {}
    for filename in train_files:
        tokens = set(newsgroup_reader(filename))
        for token in tokens:
            if token not in fid_count:
                fid_count[token] = 0
            fid_count[token] += 1

    featuremap = {}
    num_feature = 0
    for k, v in fid_count.items():
        #if v < 50:
        #    continue
        featuremap[k] = num_feature
        num_feature += 1

    return featuremap

def create_featurefile(train_files, label2id):
    featuremap = get_featuremap(train_files)
    labeltree, num_label = get_labeltree(len(label2id))

    f = open(featurefile, "w")
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

def record(writer, fn, label, featuremap):
    tokens = set(newsgroup_reader(fn))
    #tokens = newsgroup_reader(fn)
    tokens = [t for t in tokens if t in featuremap]
    if len(tokens) == 0:
        return;
    output = str(label) + " "
    for token in tokens:
        output += str(featuremap[token]) + " "
    writer.write(output[:-1] + "\n")


def data_transformer():
    label2files = {}

    label2id   = get_label2id()
    g = os.walk(corpus)
    for path,d,filelist in g:
        for filename in filelist:
            fullfn = os.path.join(path, filename)
            labelid = get_labelid_from_filename(fullfn, label2id)
            if labelid < 0:
                continue
            if labelid not in label2files:
                label2files[labelid] = []
            label2files[labelid].append(fullfn)

    train_files, val_files, test_files = [], [], []
    for k, v in label2files.items():
        random.shuffle(v)
        train_files.extend(v[:int(len(v)*0.7)])
        val_files.extend(v[int(len(v)*0.7):int(len(v)*0.9)])
        test_files.extend(v[int(len(v)*0.9):])

    featuremap, label2id = create_featurefile(train_files, label2id)
    trainwriter = open(trainfile, "w")
    valwriter   = open(valfile, "w")
    testwriter  = open(testfile, "w")
    for f in train_files:
        labelid = get_labelid_from_filename(f, label2id)
        record(trainwriter, f, labelid, featuremap)
    for f in val_files:
        labelid = get_labelid_from_filename(f, label2id)
        record(valwriter, f, labelid, featuremap)
    for f in test_files:
        labelid = get_labelid_from_filename(f, label2id)
        record(testwriter, f, labelid, featuremap)

    trainwriter.close()
    valwriter.close()
    testwriter.close()

data_transformer()

