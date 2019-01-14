#! /usr/bin/python3
import numpy as np
import pandas as pd

index_dic = {}
feat_list = ['AdID', 'DisplayURL', 'AdvertiserID', 'KeywordID', 'TitleID', 'DescriptionID']

if __name__ == '__main__':
    df = pd.read_csv("ad.filter.csv")
    of = open("ad.gby.csv", 'w')
    of.write("{}\n".format(",".join(feat_list)))

    cnt = 0
    of2 = open("AdID.map", 'w')
    group = df.groupby(by='AdID')
    for adid, ad_df in group:
        output = '{}'.format(adid)
        of2.write("{} {}\n".format(cnt, adid))
        cnt += 1
        for feat in feat_list[1:]:
            output = "{},".format(output)
            if feat in ad_df:
                attr_list = ad_df.loc[:,feat].unique()
                attr_str = "|".join(map(str, attr_list))
                output = "{}{}".format(output,attr_str)
        output = "{}\n".format(output)
        of.write(output)
