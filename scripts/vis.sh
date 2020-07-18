#!/bin/bash

echo "Visualization Single methods...";


for root in VC24; do
    for ver in RGB2LAB; do
        for net in res18ynetsync; do
            python src/vis.py -folders $net-$root-$ver -root $root;
        done
    done
done

echo "Visualization comparison methods...";


# for root in VC24; do
#     for baseline in pix2pixHD-VC24; do
#         for our in res18ynetsync-VC24-RGB2LAB; do
#             python src/vis.py -folders $baseline $our -root $root -label True;
#         done
#     done
# done

echo 'DONE'
