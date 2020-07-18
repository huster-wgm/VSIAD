#!/bin/bash

echo "Evaluate models ...";

#res18ynetsync-VC24-RGB2LAB
for root in VC24; do
    for ver in L2LAB; do
        for net in res18ynetsync; do
            python src/test.py -checkpoints $net-$root-$ver\_iter_100000.pth;
        done
    done
done

echo 'DONE'
