#!/bin/bash

case $1 in
  0)
    # Ext & logs_pth
    ext=mf-mf-ns
    logs_pth=logs/${ext}
    # Data
    tr=ob.tr.mf.ffm
    te=ob.te.sub.mf.ffm
    item=item.mf.ffm
    # Var
    k=128
    ns='--ns'
    t=46
    ;;
  1)
    # Ext & logs_pth
    ext=mf-ffm-ns
    logs_pth=logs/${ext}
    # Data
    tr=ob.tr.mf.ffm
    te=ob.te.sub.mf.ffm
    item=item.ffm
    # Var
    k=128
    ns='--ns'
    t=46
    ;;
  2)
    # Ext & logs_pth
    ext=mf-ffm
    logs_pth=logs/${ext}
    # Data
    tr=ob.tr.mf.ffm
    te=ob.te.sub.mf.ffm
    item=item.ffm
    # Var
    k=128
    ns=''
    t=46
    ;;
  3)
    # Ext & logs_pth
    ext=ffm-ffm
    # Data
    name=kdd12
    tr=user.tr.ffm
    te=user.va.ffm
    item=ad.ffm
    # Var
    k=16
    ns=''
    t=200
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  4)
    # Ext & logs_pth
    ext=fm-fm
    # Data
    name=kdd12
    tr=user.tr.fm
    te=user.va.fm
    item=ad.fm
    # Var
    k=16
    ns=''
    t=200
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  *)
    echo "No match"
    exit
esac

# Create logs_pth
echo "Do $ext"
mkdir -p $logs_pth

# 2^-6, 2^-7
# w in 0.015625 0.0078125
# 2^-8~-11
# w in 0.00390625 0.001953125 0.0009765625 0.00048828125
# l in 0.25 1 4 16
task(){
  for w in 0.00390625 0.001953125
  do
      for l in 4
      do
        echo "./train -k $k -l $l -t ${t} -r -1 -w $w $ns -c 12 -p ${te} ${item} ${tr} > $logs_pth/${tr}.$l.$w.${ext}"
      done
  done
}

# Number of parameter set do in once.
echo -n "Number of param run at once: "
read num_core
task | xargs -d '\n' -P $num_core -I {} sh -c {} &
