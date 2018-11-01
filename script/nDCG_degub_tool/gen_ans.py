#! /usr/bin/python3

import math
def score( i ):
    return 1.0 / math.log2(i + 2)

def idcg( k ):
    res = 0.0
    for i in range(k):
        res += score(i)
    return res

def dcg( label, rank, k):
    res = 0
    i = 0
    for x in rank:
        if x in label:
            res += score(i)
        i += 1
        if i == k:
            break
    return res

def ndcg( label, rank, k ):
    l = min(len(label), k)
    return dcg(label, rank, k) / idcg(l)

rf = open('case1.mf', 'r')
of = open('ans.txt', 'w')
rank = [0,1,2,3,4,5,6,7,8,9]
k = 5

def go():
    rf.seek(0)
    for line in rf:
        label = line.strip().split()[0].split(',')
        label = list(map(int, label))
        score = ndcg(label, rank, k)
        of.write("{:.4f}\n".format(score))
        print("{:.4f}".format(score))

go()
