#!/bin/bash

send(){
    dir=$1
    server=$2
    for file in $1/*
    do
        rsync -ai $file $server:/tmp2/b02701216/libffm_vs_ocffm/models& 
    done
}


send ob.trva.ffm.1e-3.64.20.0.003125 linux1
send ob.trva.ffm.1e-3.64.20.0.0125 linux2
send ob.trva.ffm.1e-3.64.20.0.05 linux3
send ob.trva.ffm.1e-4.64.20.0.003125 linux4
send ob.trva.ffm.1e-4.64.20.0.0125 linux5
send ob.trva.ffm.1e-4.64.20.0.05 linux6
send ob.trva.ffm.1e-5.64.20.0.003125 linux7
send ob.trva.ffm.1e-5.64.20.0.0125 linux8
send ob.trva.ffm.1e-5.64.20.0.05 linux9
send ob.trva.ffm.1e-6.64.20.0.003125 linux10
send ob.trva.ffm.1e-6.64.20.0.0125 linux11
send ob.trva.ffm.1e-6.64.20.0.05 linux12

