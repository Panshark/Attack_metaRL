from datetime import datetime
import gym
import torch
import json
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from maml_rl.utils import global_tensor_val

from tensorboardX import SummaryWriter


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)
    writer = SummaryWriter(summary_file_path + datetime.now().strftime("%y-%m-%d-%H-%M"))
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))# Create enviroment
    env.close()

    # Attacker
    global_tensor_val._init()
    global_tensor_val.set_tensor_value('Adv', config['Adv'])

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

    Adv=torch.tensor(config['Adv'], requires_grad=False)
    print("Initial Adv is ", Adv)
    
    epoch = 1.0
    global_episode = 1
    global_episode_val = 0
    lr = 0.1
    
    for batch in trange(config['num-batches']):# progress bar with training process
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr']*Adv.item(),
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        # futures = sampler.sample_async(tasks,
        #                                num_steps=config['num-steps'],
        #                                fast_lr=lr,
        #                                gamma=config['gamma'],
        #                                gae_lambda=config['gae-lambda'],
        #                                device=args.device)
        
        lr = 0.05
        
        logs, global_episode, global_episode_val, _ = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'],
                                Adv=Adv,
                                Outer_loop=config['outer_loop'],
                                Epoch=epoch,
                                writer = writer,
                                global_episode = global_episode,
                                global_episode_val = global_episode_val)

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        writer.add_scalar("reward/sample_train", get_returns(train_episodes[0]).mean(), epoch)
        writer.add_scalar("reward/sample_valid", get_returns(valid_episodes).mean(), epoch)
        
        logs['sample_train'] = get_returns(train_episodes[0]).mean().item()
        logs['sample_valid'] = get_returns(valid_episodes).mean().item()

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)
                
        if args.output_folder is not None:
            with open(log_filename, 'a') as f:
                json.dump(logs, f, indent=2)
            f.close()
        epoch += 1


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
