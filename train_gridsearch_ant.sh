#!/bin/bash
b=1
s=0
for((i=1;i<=500;i=i+1))
   do
        k=$s
        s=$(echo "scale=3;$i/$b"|bc)
        echo $i
    #    sed -i "s/fast-lr: $k/fast-lr: $s/g" configs/maml/ant-dir.yaml
       python train_ant.py --config configs/maml/ant-dir.yaml --output-folder "ant-dir_gridsearch_seed/dir_$i" --seed $i --num-workers 8 --use-cuda
       python test.py --config "ant-dir_gridsearch_seed/dir_$i/config.json" --policy "ant-dir_gridsearch_seed/dir_$i/policy.th" --output "ant-dir_gridsearch_seed/dir_$i/results.npz" --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
       python draw.py --resultpath "ant-dir_gridsearch_seed/dir_$i" --output_folder "ant-dir_gridsearch_seed/results" --num-batches 10 --num-traj 20 
   done