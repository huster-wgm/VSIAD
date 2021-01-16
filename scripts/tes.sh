#!/bin/bash
read -p "Please Specified the Checkpoints ?= " checks;
for check in $checks; do
    echo "Evaluating => $check";
    python src/test.py -checkpoints $check;
done

echo 'DONE'
