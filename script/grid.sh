#!/bin/bash

case $1 in
  0)
    # Data
    tr=ob.tr.mf.ffm
    te=ob.te.sub.mf.ffm
    item=item.mf.ffm
    # Var
    k=128
    # extension & logs_pth
    ext=mf-mf-ns
    logs_pth=logs/${ext}
    ;;
  *)
    echo "No match"
    exit
esac

# Create logs_pth
mkdir -p logs_pth

# w in 0.00390625 0.001953125 0.0009765625 0.00048828125
# l in 0.25 1 4 16
task(){
  for w in 0.00390625 0.001953125 0.0009765625 0.00048828125
  do
      for l in 4
      do
        echo "./train -k $k -l $l -t 46 -r -1 -w $w --ns -c 12 -p ${te} ${item} ${tr} > $logs_pth/${tr}.$l.$w.${ext}"
      done
  done
}

# Number of parameter set do in once.
num_core=1
task | xargs -d '\n' -P $num_core -I {} sh -c {} &
