import re
import math
import csv
from RankingFeatures import RetrievalModel

# UNKNOWN_TOKEN = '<UNK>'
# PAD_TOKEN = '<PAD>'
# vocab_dict = dict()
# vof=open('/data/disk2/private/qiaoyf/MSMARCO/vocab.txt',mode='r',encoding='utf-8')
# for line in vof:
#     word = line.strip('\n')
#     wd = word.split(' ')[0]
#     id = int(word.split(' ')[1])
#     vocab_dict[wd] = id
# vocab_dict[UNKNOWN_TOKEN] = 0
# vocab_dict[PAD_TOKEN] = 0

def term2lm(terms):
    lm = {}
    for term in terms:
        if term in lm:
            lm[term] += 1
        else:
            lm[term] = 1
    return lm

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

def tokenize(s, length):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()[:length]
    # return [str(vocab_dict[word]) if word in vocab_dict else '0' for word in lst]
    # lst = s.split(',')
    return lst

# K = 1.2
# B = 0.75
# AVGDL = 100

f = open('/data/disk2/private/qiaoyf/MSMARCO/new_vocab.txt')
vocab = {}
for line in f:
    line = line.strip('\n').split('\t')
    vocab[line[0]] = line[1]

f = open('/data/disk2/private/qiaoyf/MSMARCOV2/Ranking/Baselines/scripts/df.tsv')
df_dic = {}
N = 0
TOT_LENGTH = 0
for idx, line in enumerate(f):

    if idx == 0:
        line = line.strip().split('\t')
        N = int(line[0])
        TOT_LENGTH = int(line[1])
        continue
    line = line.strip().split('\t')
    # if line[0] in vocab_dict:
    #     idf_dic[str(vocab_dict[line[0]])] = float(line[1])
    # if line[0] in vocab:
    df_dic[line[0]] = float(line[1])

qrels = {}
with open('/data/disk1/private/zhangjuexiao/MSMARCOReranking/qrels.dev.tsv', mode='r', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        qid = row[0]
        did = row[2]
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(did)

# f = open('/data/disk1/private/zhangjuexiao/MSMARCOReranking/rawdata/top1000.dev.tsv')
# doc_set = {}
# for line in f:
#     line = line.strip().split('\t')

#     qid = line[0]
#     did = line[1]

#     query = line[2]
#     doc = line[3]

#     if qid not in doc_set:
#         doc_set[qid] = []

#     doc_set[qid].append(doc)


# solved_idf = {}
# def getidf(qid, qword):
#     if qid in solved_idf:
#         if qword in solved_idf[qid]:
#             return solved_idf[qid][qword]
#         else:
#             return 0

#     solved_idf[qid] = {}
#     df = {}
#     n = 0
#     for doc in doc_set[qid]:
#         n += 1
#         doc = tokenize(doc, 200)
#         for word in set(doc):
#             if word in solved_idf[qid]:
#                 solved_idf[qid][word] += 1
#             else:
#                 solved_idf[qid][word] = 1
#     if n == 1:
#         return 0

#     for word in solved_idf[qid]:
#         solved_idf[qid][word] = math.log((n - solved_idf[qid][word] + 0.5) / (solved_idf[qid][word] + 0.5)) / math.log(n)

#     if qword in solved_idf[qid]:
#         return solved_idf[qid][qword]
#     else:
#         return 0

# def bm25(qid, query, doc):
#     score = 0.0
#     for word in query:
#         # if word not in idf_dic:
#         #     continue
#         idf = getidf(qid, word) # idf_dic[word]
#         tf = doc.count(word)

#         score += idf * tf * (K+1) / (tf + K * (1 - B + B * len(word) / AVGDL))

#     return score

'''
# f = open('/data/disk1/private/zhangjuexiao/MSMARCOReranking/rawdata/top1000.dev.tsv')
f = open('/data/disk3/private/qiaoyf/MSMARCO/top1000.dev.tsv')
tot_set = {}
for idx, line in enumerate(f):
    if idx % 10000 == 0:
        print(idx)
    line = line.strip().split('\t')

    qid = line[0]
    did = line[1]

    query = line[2]
    doc = line[3]

    # query = tokenize(query, 20)
    # doc = tokenize(doc, 200)

    if qid not in tot_set:
        tot_set[qid] = {'query': query, 'doc': []}

    tot_set[qid]['doc'].append((did, doc))
'''

retrieval_model = RetrievalModel()

def bm25_score(tot_set):
    #fout = open('/data/disk3/private/qiaoyf/MSMARCO/dev_svm.txt', 'w')
    #fake_id = 0
    #for qid in tot_set:
        #fake_id += 1
    query = tokenize(tot_set['query'], 50)

    scored_doc = []
    for idx, doc in enumerate(tot_set['doc']):# toby change
        doc_tmp = doc
        doc = tokenize(doc, 200)
        df = {qt: df_dic[qt] if qt in df_dic else 0 for qt in query}
        retrieval_model.set_from_raw(term2lm(query), term2lm(doc), df, N, TOT_LENGTH/float(N))
        scores = retrieval_model.scores()
        # toby add
        #print(scores)
        #exit(0)
        score = scores[4][1]# bm25 score
        scored_doc.append((doc_tmp, score))
            
        #idx = 0
        #score_str = ""
        # print(scores)
        #for name, score in scores:
        #    idx += 1
        #    score_str += ' {}:{}'.format(idx, score)

        # label = '1' if did in qrels[qid] else '0'
        #label = '0'

    #fout.write(label + ' ' + 'qid:{}'.format(fake_id) + ' ' + score_str + ' # {} {}'.format(qid, did) + '\n')
    return scored_doc
