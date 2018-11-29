#! /bin/bash

logs_dir="logs"
k=64
c=20
te='ob.te.ocffm'
trva='ob.trva.ocffm'
item='ad.ocffm'

create_logs_folders(){
  for model in `ls models`
  do
    base_name=`echo $model | rev | cut -d '.' -f2- | rev`
    mkdir -p $logs_dir/$base_name
  done
}

task(){
  for model in `ls models`
  do
    base_name=`echo $model | rev | cut -d '.' -f2- | rev`
    log_path=$logs_dir/$base_name/$model
    echo "./train -k $k -c $c -m models/$model -p ${te} ${item} ${trva} > $log_path"
  done
}

create_logs_folders
task
