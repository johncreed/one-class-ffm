#!/bin/bash

send(){
    dir=$1
    server=$2
    for file in $1/*
    do
        echo $2
        rsync -ai $file $server:/tmp2/b02701216/libffm_vs_ocffm/models 
    done
}

send ob.tr.libffm.1e-3.32.20.0.000390625 linux7 
send ob.tr.libffm.1e-3.32.20.0.0015625 linux7
send ob.tr.libffm.1e-3.32.20.0.00625 linux7
send ob.tr.libffm.1e-3.32.20.0.025 linux9
send ob.tr.libffm.1e-3.32.20.0.1 linux9
send ob.tr.libffm.1e-3.32.20.0.4 linux9
send ob.tr.libffm.1e-4.32.20.0.000390625 linux10
send ob.tr.libffm.1e-4.32.20.0.0015625 linux10
send ob.tr.libffm.1e-4.32.20.0.00625 linux10
send ob.tr.libffm.1e-4.32.20.0.025 linux11
send ob.tr.libffm.1e-4.32.20.0.1 linux11
send ob.tr.libffm.1e-4.32.20.0.4 linux11
send ob.tr.libffm.1e-5.32.20.0.000390625 linux12
send ob.tr.libffm.1e-5.32.20.0.0015625 linux12
send ob.tr.libffm.1e-5.32.20.0.00625 linux12
send ob.tr.libffm.1e-5.32.20.0.025 linux13
send ob.tr.libffm.1e-5.32.20.0.1 linux13
send ob.tr.libffm.1e-5.32.20.0.4 linux13
send ob.tr.libffm.1e-6.32.20.0.000390625 linux14
send ob.tr.libffm.1e-6.32.20.0.0015625 linux14
send ob.tr.libffm.1e-6.32.20.0.00625 linux14
send ob.tr.libffm.1e-6.32.20.0.025 linux2
send ob.tr.libffm.1e-6.32.20.0.1 linux2
send ob.tr.libffm.1e-6.32.20.0.4 linux2
