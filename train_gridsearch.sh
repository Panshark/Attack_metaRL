#!/bin/bash
b=100
s=0
for((i=1;i<=200;i=i+1))
   do
       k=$s
       s=$(echo "scale=3;$i/$b"|bc)
        echo $s
       sed -i "s/Adv: $k/Adv: $s/g" /home/larx/Haozhe_Lei/Attack_sgmrl/configs/maml/2d-navigation.yaml
       python train.py --config configs/maml/2d-navigation.yaml --output-folder "2d-navigation_gridsearch/dir_$s" --seed 1 --num-workers 8 --use-cuda
       python test.py --config "2d-navigation_gridsearch/dir_$s/config.json" --policy "2d-navigation_gridsearch/dir_$s/policy.th" --output "2d-navigation_gridsearch/dir_$s/results.npz" --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
       python draw.py --resultpath "2d-navigation_gridsearch/dir_$s" --output_folder "2d-navigation_gridsearch/results" --num-batches 10 --num-traj 20 
   done