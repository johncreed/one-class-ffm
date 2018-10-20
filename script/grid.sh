#!/bin/bash

case $1 in
  0)
    # Ext & logs_pth
    ext=ffm-ffm
    # Data
    name=ob
    tr=ob.tr.ffm
    te=ob.va.sub.ffm
    item=item.ffm
    # Var
    k=64
    ns=''
    t=70
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  1)
    # Ext & logs_pth
    ext=fm-fm
    # Data
    name=ob
    tr=ob.tr.fm
    te=ob.va.sub.fm
    item=item.fm
    # Var
    k=128
    ns=''
    t=70
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  2)
    # Ext & logs_pth
    ext=mf-mf
    # Data
    name=ob
    tr=ob.tr.mf
    te=ob.va.sub.mf
    item=item.mf
    # Var
    k=128
    ns=''
    t=70
    logs_pth=logs/${name}.${k}/${ext}
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
    k=32
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
    k=64
    ns=''
    t=200
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  5)
    # Ext & logs_pth
    ext=mf-mf
    # Data
    name=kdd12
    tr=user.tr.mf
    te=user.va.mf
    item=ad.mf
    # Var
    k=128
    ns=''
    t=200
    logs_pth=logs/${name}.${k}/${ext}
    ;;
  *)
    echo "No match"
    exit
esac

# 2^0 ~ 2^-11
w_all=(1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125)

choose_w_list(){
  for i in ${!w_all[@]};
  do
    printf "%s: %s(2^-%s)\n" "$i" "${w_all[$i]}" "$i"
  done
  w_train=()
  while true;
  do
    echo -n "select idx: "
    read idx
    if [ $idx -eq -1 ]
    then
      break
    fi
    w_train+=(${w_all[${idx}]})
  done

  echo -n "w choosed: "
  echo ${w_train[@]}
}


# l in 0.25 1 4 16
task(){
  for w in ${w_train[@]} 
  do
      for l in 4
      do
        echo "./train -k $k -l $l -t ${t} -r -1 -w $w $ns -c 12 -p ${te} ${item} ${tr} > $logs_pth/${tr}.$l.$w.${ext}"
      done
  done
}

grid(){
  
  # Create logs_pth
  echo "===Data: $name solve: $ext==="
  mkdir -p $logs_pth
  echo "++++++++++++++++++++++++++"
  
  # Set w range
  echo "===Set w range, -1 to exit==="
  choose_w_list
  echo "++++++++++++++++++++++++++"
  
  # Number of parameter set do in once.
  echo -n "Number of param run at once: "
  read num_core
  echo "++++++++++++++++++++++++++"
  
  # Check the right command
  echo "===All cmd to run==="
  task
  echo -n "Start ? [y/n]"
  read std
  echo "++++++++++++++++++++++++++"
  if [[ $std =~ y ]]
  then
    echo "run"
    task | xargs -d '\n' -P $num_core -I {} sh -c {} &
  else
    echo "no run"
  fi
}

grid
