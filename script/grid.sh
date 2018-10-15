data_tr=ob.tr.mf.ffm
data_te=ob.te.sub.mf.ffm
item=item.mf.ffm
k=128
for w in 0.00390625 0.001953125 0.0009765625 0.00048828125
do
    for l in 4
    do
        #./train -k $k -l $l -t 41 -r -1 -w $w -c 12 -p ${data_te} ${item} ${data_tr} > logs/${data_tr}.$l.$w
        ./train -k $k -l $l -t 46 -r -1 -w $w --ns -c 12 -p ${data_te} ${item} ${data_tr} > logs/${data_tr}.$l.$w.ns
    done
done
