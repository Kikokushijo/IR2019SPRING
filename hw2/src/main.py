import os
from collections import defaultdict

# filename
DATA_DIR = '../data'
TRAIN_ANS_FILE = os.path.join(DATA_DIR, 'ans_train.csv')
QUERY_TRAIN_FILE = os.path.join(DATA_DIR, 'query_train.xml')
QUERY_TEST_FILE = os.path.join(DATA_DIR, 'query_test.xml')
FILE_LIST = os.path.join(DATA_DIR, 'file-list')
INV_FILE = os.path.join(DATA_DIR, 'inverted-file')
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab.all')

# building maps
docname2id = dict()
with open(FILE_LIST, 'r') as f:
    for idx, line in enumerate(f):
        docname2id[line.strip()] = idx

word2id = dict()
with open(VOCAB_FILE, 'r') as f:
    f.readline()
    for idx, line in enumerate(f, 1):
        word2id[line.strip()] = idx

word_freq = dict()
with open(INV_FILE, 'r') as f:
    while True:
        line = f.readline().strip()
        if not line:
            break
        id_1, id_2, doc_count = [int(i) for i in line.split(' ')]
        doc_records = defaultdict(int)
        for i in range(doc_count):
            doc_id, freq = [int(i) for i in f.readline().strip().split(' ')]
            doc_records[doc_id] = freq
        word_freq[(id_1, id_2)] = doc_records
print(word_freq[(1, -1)])
