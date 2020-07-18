#!/bin/bash

read -p "Specified Device([0, 1, 2, 3]) = " device;


# # no perceptual
# for root in VC24; do
#     for ver in RGB2LAB; do
#         for net in res18ynet res18ynetsync; do
#             CUDA_VISIBLE_DEVICES=$device python src/evaluate.py -dir $net-$root-$ver-nop;
#         done
#     done
# done

# # has perceptual
# for root in VC24; do
#     for ver in RGB2LAB; do
#         for net in res18ynet res18ynetsync; do
#             CUDA_VISIBLE_DEVICES=$device python src/evaluate.py -dir $net-$root-$ver;
#         done
#     done
# done

# # color space
# for root in VC24; do
#     for ver in L2LAB LAB2LAB; do
#         for net in res18ynetsync; do
#             CUDA_VISIBLE_DEVICES=$device python src/evaluate.py -dir $net-$root-$ver;
#         done
#     done
# done

echo "DONE"
