#!/bin/bash
b=100
s=0
for((i=1;i<=200;i=i+1))
   do
       k=$s
       s=$(echo "scale=3;$i/$b"|bc)
        echo $s
       sed -i "s/Adv: $k/Adv: $s/g" /home/larx/Haozhe_Lei/Attack_sgmrl/configs/maml/halfcheetah-vel.yaml
       python train_copy.py --config configs/maml/halfcheetah-vel.yaml --output-folder "halfcheetah-vel_gridsearch/dir_$s" --seed 1 --num-workers 8 --use-cuda
       python test.py --config "halfcheetah-vel_gridsearch/dir_$s/config.json" --policy "halfcheetah-vel_gridsearch/dir_$s/policy.th" --output "halfcheetah-vel_gridsearch/dir_$s/results.npz" --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
       python draw.py --resultpath "halfcheetah-vel_gridsearch/dir_$s" --output_folder "halfcheetah-vel_gridsearch/results" --num-batches 10 --num-traj 20 
   done