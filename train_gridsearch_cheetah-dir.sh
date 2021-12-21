#!/bin/bash
b=100
s=0
for((i=1;i<=50;i=i+1))
   do
       k=$s
       s=$(echo "scale=3;$i/$b"|bc)
        echo $s
       sed -i "s/fast-lr: $k/fast-lr: $s/g" configs/maml/halfcheetah-dir.yaml
       python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder "halfcheetah-dir_gridsearch/dir_$s" --seed 1 --num-workers 8 --use-cuda
       python test.py --config "halfcheetah-dir_gridsearch/dir_$s/config.json" --policy "halfcheetah-dir_gridsearch/dir_$s/policy.th" --output "halfcheetah-dir_gridsearch/dir_$s/results.npz" --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
       python draw.py --resultpath "halfcheetah-dir_gridsearch/dir_$s" --output_folder "halfcheetah-dir_gridsearch/results" --num-batches 10 --num-traj 20 
   done