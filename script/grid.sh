#!/bin/bash

list_solve_type(){
echo "=== Choose solve type ===
0: ob, ffm-ffm
1: ob, fm-fm
2: ob, mf-mf-ns
3: kdd, ffm-ffm
4: kdd, fm-fm
5: kdd, mf-mf--ns"
echo -n "sovle (0~5): "
read solve_type
}


set_up_solve_type(){
  echo -n "latent vector size(k): "
  read k

  case $solve_type in
    0)
      # Ext & logs_pth
      tr_ext=ffm
      i_ext=ffm
      # Data
      name=ob
      tr=ob.tr.${tr_ext}
      te=ob.va.sub.${tr_ext}
      item=item.${i_ext}
      # Var
      ns=''
      t=70
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    1)
      # Ext & logs_pth
      tr_ext=fm
      i_ext=fm
      # Data
      name=ob
      tr=ob.tr.${tr_ext}
      te=ob.va.sub.${tr_ext}
      item=item.${i_ext}
      # Var
      ns=''
      t=70
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    2)
      # Ext & logs_pth
      tr_ext=mf
      i_ext=mf
      # Data
      name=ob
      tr=ob.tr.${tr_ext}
      te=ob.va.sub.${tr_ext}
      item=item.${i_ext}
      # Var
      ns='--ns'
      t=70
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    3)
      # Ext & logs_pth
      tr_ext=ffm
      i_ext=ffm
      # Data
      name=kdd12
      tr=user.tr.${tr_ext}
      te=user.va.${tr_ext}
      item=ad.${i_ext}
      # Var
      ns=''
      t=200
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    4)
      # Ext & logs_pth
      tr_ext=fm
      i_ext=fm
      # Data
      name=kdd12
      tr=user.tr.${tr_ext}
      te=user.va.${tr_ext}
      item=ad.${i_ext}
      # Var
      ns=''
      t=200
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    5)
      # Ext & logs_pth
      tr_ext=mf
      i_ext=mf
      # Data
      name=kdd12
      tr=user.tr.${tr_ext}
      te=user.va.${tr_ext}
      item=ad.${i_ext}
      # Var
      ns='--ns'
      t=200
      ext=${tr_ext}-${i_ext}${ns}
      ;;
    *)
      echo "No match"
      exit
  esac
  logs_pth=logs/${name}.${k}/${ext}
}


choose_w_list(){
# 2^0 ~ 2^-11
w_all=(1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125)
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

  echo -n "w list = [ "
  echo ${w_train[@]}]
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

list_param(){
echo "===Parameter list==="
echo "data: $name
tr: $tr
va: $te
item: $item
k: $k
t: $t
ns: $ns
w list: ${w_train[@]}
log_path: ${logs_pth}"
}

grid(){

  # Choose solve type
  clear
  list_solve_type
  set_up_solve_type
  
  # Set w range
  clear
  echo "===Set w range, -1 to exit==="
  choose_w_list
  echo "++++++++++++++++++++++++++"
  
  # logs_pth
  
  # Check the right command
  clear
  list_param
  echo "===All cmd to run==="
  task

  # Number of parameter set do in once.
  echo -n "Number of param run at once: "
  read num_core
  echo "++++++++++++++++++++++++++"

  # Start or not
  echo -n "Start ? [y/n]"
  read std
  echo "++++++++++++++++++++++++++"
  if [[ $std =~ y ]]
  then
    # Create logs_pth
    mkdir -p $logs_pth
    echo "run"
    task | xargs -d '\n' -P $num_core -I {} sh -c {} &
  else
    echo "no run"
  fi
}

grid
