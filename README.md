## Attack on Meta-Learning Pytorch Implementation

Original implementation by Tristan Deleu https://github.com/tristandeleu/pytorch-maml-rl.

![HalfCheetahDir](https://github.com/Panshark/Attack_metaRL/blob/main/image_source/withoutattack_cheetah.gif)

Usage:

Train:

```
python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation --seed 1 --num-workers 8 --use-cuda
```
Test:

```
python test.py --config maml-halfcheetah-vel/config.json --policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
```

Draw:

```
python draw.py --resultpath maml-halfcheetah-vel --output_folder maml-halfcheetah-vel/returns --num-batches 10 --num-traj 20 
```

2D-Draw:

```
python draw_2d.py --config 2d-navigation/config.json --policy 2d-navigation/policy.th --output 2d-navigation/record --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
```

