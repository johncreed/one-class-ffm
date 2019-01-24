#! /usr/bin/python3
import numpy
import pandas
import sys

threshold =int( input("Enter threshold: "))
type(threshold)
df = pandas.read_csv("training.csv")

# Filter by click and UserId != 0
df_pos=df.loc[df['Click'] > 0, :]
df_pos_nozero = df_pos.loc[df['UserID'] != 0, :]
df_pos_nozero.to_csv("tmp")

# Filter by top clicked ad
ad_list=df_pos_nozero.AdID.value_counts()
ad_list=ad_list.sort_values(ascending=False)
num = ad_list.where(ad_list > threshold).count()
print(num)
ad_filter=ad_list.iloc[0:num]
df_filter=df_pos_nozero.loc[df_pos_nozero['AdID'].isin(ad_filter.index)]

# Ad filter csv
df_ad=df_filter[['AdID','DisplayURL','AdvertiserID', 'KeywordID', 'TitleID', 'DescriptionID' ]]
df_ad.to_csv("ad.filter.csv", index=False)

# User filter csv
df_user=df_filter[['AdID','UserID', 'QueryID', 'Depth']]
df_user.to_csv("user.filter.csv", index=False)
