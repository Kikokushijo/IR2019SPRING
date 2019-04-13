import os
import csv
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
from collections import Counter
from argparse import ArgumentParser
from scipy.sparse import dok_matrix

def two_char_to_gramid(c1, c2):
    try:
        c1 = word2id[c1]
        if c2 is None:
            c2 = -1
        else:
            c2 = word2id[c2]
        return gram2id[(c1, c2)]
    except:
        return 0

def generate_candidates_weight(k_1=1.5, b=0.75):
    weight = dok_matrix((doc_num, word_num), dtype=np.float32)
    for idx, (gram, records) in enumerate(gram_freq.items(), 1):
        word_id = gram2id[gram]
        IDF = np.log((doc_num - len(records) + 0.5) / (len(records) + 0.5))
        for doc_id, freq in records.items():
            TF = (k_1 + 1) * freq / (freq + k_1 * (1 - b + b * doc2len[doc_id] / avdl))
            weight[doc_id, word_id] = TF * IDF
        if idx % 1000 == 0:
            print('Has indexed %d grams' % idx)
    return weight

def generate_queries_weight(ka=100):
    queries = np.zeros((query_num, word_num))
    for query_id, child in enumerate(root):
        query = child[4].text.strip('\n。 ').split('、')
        grams = []
        for w in query:
            if len(w) == 1:
                grams.append(two_char_to_gramid(w, None))
                break
            else:
                for c in w:
                    grams.append(two_char_to_gramid(c, None))
                for ci, cj in zip(w[:-1], w[1:]):
                    grams.append(two_char_to_gramid(ci, cj))
        for word, freq in Counter(grams).items():
            if not word:
                continue
            TF = (ka + 1) * freq / (freq + ka)
            queries[query_id, word] = TF
    return queries

def MAP(top100):
    AP = []
    with open(TRAIN_ANS_FILE, 'r') as f:
        f.readline()
        for line, rank in zip(f, top100):
            idx, answer = line.strip().split(',')
            answer = set(answer.split())
            rank = [(id2docname[i].split('/')[-1].lower()) for i in rank]
            hit = 0
            P = []
            for rank_i, rank in enumerate(rank, 1):
                if rank in answer:
                    hit += 1
                    P.append(hit / rank_i)
            AP.append(sum(P) / len(P))
    return sum(AP) / len(AP)

# set parser
parser = ArgumentParser()
parser.add_argument(
    "-r", action="store_true",
    help="Turn on the relevance feedback", 
    dest="rel_switch", default=False
)
parser.add_argument(
    "-b", action="store_true",
    help="Run the best version", 
    dest="best_switch", default=False
)
parser.add_argument(
    "-i", action="store",
    help="Filename of input query file", 
    dest="query_file", default="data/query-test.xml"
)
parser.add_argument(
    "-o", action="store",
    help="Filename of output ranked list file", 
    dest="ranked_list", default='output.csv'
)
parser.add_argument(
    "-m", action="store",
    help="Path of input model directory", 
    dest="model_dir", default="data"
)
parser.add_argument(
    "-d", action="store",
    help="Path of NTCIR documents directory", 
    dest="NTCIR_dir", default="data/CIRB010"
)
args = parser.parse_args()

# set filename / pathname
MODEL_DIR = args.model_dir
DOC_DIR = args.NTCIR_dir
QUERY_TEST_FILE = args.query_file
FILE_LIST = os.path.join(MODEL_DIR, 'file-list')
INV_FILE = os.path.join(MODEL_DIR, 'inverted-file')
VOCAB_FILE = os.path.join(MODEL_DIR, 'vocab.all')
OUTPUT_FILE = args.ranked_list

# building maps
docname2id = dict()
with open(FILE_LIST, 'r') as f:
    for idx, line in enumerate(f):
        docname2id[line.strip()] = idx
id2docname = {y:x for x, y in docname2id.items()}

word2id = dict()
with open(VOCAB_FILE, 'r') as f:
    f.readline()
    for idx, line in enumerate(f, 1):
        word2id[line.strip()] = idx

word_freq = dict()
gram_freq = dict()
gram2id = dict()
with open(INV_FILE, 'r') as f:
    idx = 0
    while True:
        line = f.readline().strip()
        if not line:
            break
        id_1, id_2, doc_count = [int(i) for i in line.split(' ')]
        doc_records = defaultdict(int)
        for i in range(doc_count):
            doc_id, freq = [int(i) for i in f.readline().strip().split(' ')]
            doc_records[doc_id] = freq
        if id_2 == -1:
            word_freq[(id_1)] = doc_records
        gram_freq[(id_1, id_2)] = doc_records
        gram2id[(id_1, id_2)] = idx
        idx += 1

# calculate doc length
doc2len = defaultdict(int)
for docname, idx in docname2id.items():
    docname = os.path.join(DOC_DIR, '/'.join(docname.split('/')[1:]))
    doc2len[idx] = os.path.getsize(docname)
avdl = sum([length for _, length in doc2len.items()]) / len(doc2len)

# Testing
tree = ET.ElementTree(file=QUERY_TEST_FILE)
root = tree.getroot()
query_num = len(root)
word_num = len(gram2id) + 1
doc_num = len(docname2id)

candidates = generate_candidates_weight(k_1=1.6, b=0.75)
queries = generate_queries_weight(ka=5.0)
ret = (np.transpose(candidates.dot(np.transpose(queries)))).argsort(axis=1)[:, ::-1][:, :100]

if args.best_switch:
    pass
elif args.rel_switch:
    new_queries = np.zeros((query_num, word_num))
    for idx, (r, q) in enumerate(zip(ret, queries)):
        new_queries[idx, :] = q + 0.75 * np.mean(candidates[r[:10]], axis=0)
    ret = (np.transpose(candidates.dot(np.transpose(new_queries)))).argsort(axis=1)[:, ::-1][:, :100]
else:
    pass

with open(OUTPUT_FILE, 'w+') as f:
    csvwriter = csv.writer(f, delimiter=',')
    csvwriter.writerow(['query_id', 'retrieved_docs'])
    for idx, result in enumerate(ret, 11):
        csvwriter.writerow([str(idx).zfill(3), ' '.join([(id2docname[i].split('/')[-1].lower()) for i in result])])