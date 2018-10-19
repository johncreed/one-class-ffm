#! /usr/bin/python3
import pandas as pd
import numpy as np

dfa = pd.read_csv("ad_filter.csv")
dfe = pd.read_csv("events_filter.csv")
dfc = pd.read_csv("click_filter.csv")
dfm = pd.read_csv("documents_meta.csv")
dfa['label'] = dfa.index.to_series()
dfe = dfe.merge(dfc, left_on='display_id', right_on='display_id', how='left')
dfe = dfe.merge(dfa, left_on='ad_id', right_on='ad_id', how='left')
dfe = dfe.merge(dfm, left_on='document_id_x', right_on='document_id', how='left')
dfe.sort_values(by='timestamp', ascending=False)
print("Save file")
dfe.to_csv("events_filter_label.csv", index=False)

dfa = dfe.merge(dfa, left_on='document_id', right_on='document_id', how='left')
dfa.to_csv("ad_filter.csv", index=False)
