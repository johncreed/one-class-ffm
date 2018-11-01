#! /bin/bash

if [ -z "$1" ]
then
  echo "Usage: ./sort_logs.sh dir"
  exit
fi

list_logs(){
for f in $1/*
do
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo $f
  tail -n +1 $f | sort -g -k 2
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done
}

list_logs $1  | less 
