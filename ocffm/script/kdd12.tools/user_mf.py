#! /usr/bin/python3
import csv

# Field 1, 2
field0 = ['UserID']
field1 = []
s_field = field0 + field1
all_fields = [s_field]

feat_dict = {}
all_idx = [-1]


def add_feat(key, value, field):
    global all_idx
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    all_idx[field] += 1
    feat_dict[real_key] = all_idx[field]
    return all_idx[field]

def make_tuple(feat_list,field):
    feat_str = ["%d:1" % i for i in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))

of = open('user.mf', 'w')
rf = open('user.gby.csv')
for line in csv.DictReader(rf, delimiter=','):
    # Lable
    output = line['AdID'].strip().replace("|", ",")
    #Feature
    for i, field_i in enumerate(all_fields):
        feat_idx_list = []
        for key in field_i:
            if line[key] == "":
                continue
            values = line[key].split("|")
            for val in values:
                feat_idx_list.append(add_feat(key, val.strip(), i))
        output = "{} {}".format(output," ".join(make_tuple(feat_idx_list, i)))

    print( output, file=of )

print(all_idx)
