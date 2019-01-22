#!/bin/bash

num_core=20

r_train=(0.4 0.1 0.025 0.00625 0.0015625 0.000390625)
l_train=(1e-6 1e-5 1e-4 1e-3)
k=32
t=20
libffm_tr="ob.tr.libffm"
libffm_va="ob.va.libffm"
stdout_pth="log"

create_models_pth(){
  for r in ${r_train[@]} 
  do
      for l in ${l_train[@]}
      do
        mkdir -p $models_pth/${libffm_tr}.$l.$k.$t.$r
        mkdir -p $stdout_pth
      done
  done
}

task(){
  for r in ${r_train[@]} 
  do
      for l in ${l_train[@]}
      do
        echo "./ffm-train -k $k -l $l -t $t -r $r --no-norm ${libffm_tr} $models_pth/${libffm_tr}.$l.$k.$t.$r/${libffm_tr}.$l.$k.$t.$r > $stdout_pth/${libffm_tr}.$l.$k.$t.$r"
      done
  done
}

create_models_pth
task > .task_file.tmp
cat .task_file.tmp | xargs -d '\n' -P $num_core -I {} sh -c {} &

