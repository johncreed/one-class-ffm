#! /usr/bin/python3
import numpy as np
import pandas as pd
import sys

index_dic = {}
feat_list = ['AdID', 'UserID', 'QueryID', 'Depth']

map_file = open("AdID.map", "r")
map_dict = {}
def built_map_dict():
    for line in map_file:
        token = line.strip().split()
        if token[1] in map_dict:
            sys.exit("AdID is not unique")
        map_dict[token[1]] = token[0]

def adid_map( ad_list ):
    match = lambda x : map_dict[str(x)]
    return map(match, ad_list)


if __name__ == '__main__':
    built_map_dict()

    df = pd.read_csv("user.filter.csv")
    of = open("user.gby.csv", 'w')
    of.write("{}\n".format(",".join(feat_list)))

    group = df.groupby(feat_list[1:])
    for key, ad_df in group:
        attr_list = ad_df.loc[:,'AdID'].unique()
        attr_str = "|".join( adid_map(attr_list) )
        output = "{},{}\n".format(attr_str, ",".join(map(str, key)))

        of.write(output)
