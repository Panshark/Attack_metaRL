import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
# from maml_rl.utils.drawing_tools import update_points

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    tasks = sampler.sample_tasks(num_tasks=1)
    train_episodes, valid_episodes = sampler.sample(tasks,
                                                    num_steps=config['num-steps'],
                                                    fast_lr=config['fast-lr'],
                                                    gamma=config['gamma'],
                                                    gae_lambda=config['gae-lambda'],
                                                    device=args.device)
    
    b = to_numpy(valid_episodes[0].observations)
    
    def update_points(num):
        point_ani.set_data(x[num], y[num])
        return point_ani,
    for i in range(len(b)):
        c = b[i]
        x = np.insert(c[:,0],0,0)
        y = np.insert(c[:,1],0,0)
        fig = plt.figure(tight_layout=True)
        plt.plot(*list(tasks[0].values())[0], 'y*')
        plt.plot(0, 0, 'kv')
        plt.plot(x,y)
        point_ani, = plt.plot(x[0], y[0], "ro")
        plt.grid(ls="--")

        ani = animation.FuncAnimation(fig, update_points, frames = np.arange(0, 20), interval=1000, blit=True)
        # mywriter = animation.FFMpegWriter(fps=60)
        record_filename = os.path.join(args.output, 'myanimation'+str(i)+'.gif')
        # ani.save(record_filename, writer=mywriter)
        ani.save(record_filename,writer='pillow')
        # plt.show()
    
    
    
    
    b = to_numpy(valid_episodes[0].observations.view(-1, 2))
    fig, ax = plt.subplots()
    plt.plot(*list(tasks[0].values())[0], 'r*')
    x = b[:,0]
    y = b[:,1]
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'b,')

    def init():
        ax.set_xlim(min(-1,min(x)), max(1,max(x)))
        ax.set_ylim(min(-1,min(y)), max(1,max(y)))
        return ln,

    def update(num):
        xdata.append(x[num])
        ydata.append(y[num])
        ln.set_data(xdata, ydata)
        return ln,

    ani = animation.FuncAnimation(fig, update, frames = np.arange(0, len(x)),
                        init_func=init, blit=False)
    mywriter = animation.FFMpegWriter(fps=60)
    record_filename = os.path.join(args.output, 'myanimation_all.mp4')
    ani.save(record_filename,writer=mywriter)
    # ani.save("./test.gif",writer='pillow')


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')
    # parser.add_argument('--output_folder', type=str, required=True,
    #     help='path to the records file directory')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
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
