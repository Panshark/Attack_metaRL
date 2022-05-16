from datetime import datetime
import gym
import torch
import json
import os
import yaml
import time
import numpy as np
from tqdm import trange
import _pickle as pickle
import math
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from maml_rl.utils import global_tensor_val

from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)# Read yaml file and change to class 'dict'

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')
        log_filename = os.path.join(args.output_folder, 'logs.txt')
        summary_file_path = os.path.join(args.output_folder, 'tensorboard')

        log = {}
        logs = {}
        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)
    writer = SummaryWriter(summary_file_path + datetime.now().strftime("%y-%m-%d-%H-%M"))    
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # max_value = 10000
    # min_value = -10000

    # Attacker
    global_tensor_val._init()
    global_tensor_val.set_tensor_value('Adv', 1.0)
    Adv=global_tensor_val.get_value('Adv')
    print("Initial Adv is ", global_tensor_val.get_value('Adv'))

    # Optimizer
    optimizer = torch.optim.SGD([Adv], lr = config['attacker-lr'], momentum=config['attacker-momentum'], dampening=config['attacker-dampening'], weight_decay=config['attacker-weight_decay'])

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))# Create enviroment
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    if os.path.exists(policy_filename):
        print("Load exist policy!")
        with open(policy_filename, 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device(args.device))
            policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)
    
    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)# Define MAML method

    epoch = 1.0
    global_episode = 1
    global_episode_val = 0
    Adv_grads_before = 0

    stop_trigger = 0
    writer.add_scalar("attacker/Adv", to_numpy(Adv).tolist(), 0)

    for batch in trange(config['whole-batch-number-one-step']):# progress bar with training process
        # Adv_lr = config['attacker-lr']
        # logs['epoch'] = epoch
        # logs['time'] = time.asctime( time.localtime(time.time()) )
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                    num_steps=config['num-steps'],
                                    fast_lr=config['fast-lr']*Adv.item(),
                                    gamma=config['gamma'],
                                    gae_lambda=config['gae-lambda'],
                                    device=args.device)
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------
        logs, global_episode, global_episode_val, _ = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'],
                                Adv=Adv,
                                Epoch=epoch,
                                Outer_loop=config['outer_loop'],
                                writer = writer,
                                global_episode = global_episode,
                                global_episode_val = global_episode_val)
        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)
            with open(log_filename, 'a') as f:
                json.dump(logs, f, indent=2)
        
# ---------------------------------------------------------------------------------------------------------------------------------------------
        logs, Adv, optimizer, Adv_grads = metalearner.step_attacker_with_opt(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'],
                                Adv=Adv,
                                Epoch=epoch,
                                optimizer=optimizer,
                                clip_value=config['clip_value'],
                                Outer_loop=config['outer_loop'],
                                writer=writer)
    
        
        # Early stop
        # Adv_distance = abs(logs['Adv_before']-logs['Adv_after'])/abs(logs['Adv_before'])
        Adv_grads_after = Adv_grads.item()
        Adv_distance = abs(Adv_grads_after + Adv_grads_before)
        if(Adv_distance <= config['restricted-distance']):
            stop_trigger+=1
        else:
            stop_trigger=0
        Adv_grads_before = Adv_grads_after
        
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        writer.add_scalar("reward/sample_train", get_returns(train_episodes[0]).mean(), epoch)
        writer.add_scalar("reward/sample_valid", get_returns(valid_episodes).mean(), epoch)
        
        logs['sample_train'] = get_returns(train_episodes[0]).mean().item()
        logs['sample_valid'] = get_returns(valid_episodes).mean().item()
        
        if args.output_folder is not None:
            with open(log_filename, 'a') as f:
                json.dump(logs, f, indent=2)
            f.close()


        # Early stop
        if stop_trigger >= config['stop-threshold']:
            break
        


        # lr decay
        epoch += 1
        if(epoch%config['lr_decay']==0):
            optimizer.param_groups[0]['lr']=config['attacker-lr']/((epoch/config['lr_decay'])**0.5)
#----------------------------------------------------------------------------------

    print("Finall Adv is: ", Adv)   
    


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
