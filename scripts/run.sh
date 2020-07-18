#!/bin/bash

echo "Training models";

for root in VC24; do
    for ver in L2LAB RGB2LAB LAB2LAB; do
        for net in res18ynetsync; do
            python src/train.py -root $root -ver $ver -net $net -trigger iter -interval 1000 -terminal 100000 -batch_size 8;
        done
    done
done

echo "DONE"
