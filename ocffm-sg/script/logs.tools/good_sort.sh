#! /bin/bash

if [ -z "$1" ]
then
  echo "Usage: ./good_sort.sh dir"
  exit
fi

for dir in $1/*
do
  files=$dir/*
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
  echo "$dir"
  echo "pre@5"
  sort -g -k 2 $files | tail -n 1
  echo "pre@10"
  sort -g -k 3 $files | tail -n 1
  echo "pre@20"
  sort -g -k 4 $files | tail -n 1
  echo "pre@40"
  sort -g -k 5 $files | tail -n 1
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done
