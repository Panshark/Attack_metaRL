import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

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
    point = 0
    while(True):
        logs = {'tasks': []}
        train_returns, valid_returns = [], []
        for batch in trange(args.num_batches):
            P = -2.0+point*0.002
            temp_tasks = np.array([ P, P])
            tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            for i in range(len(tasks)):
                tasks[i]['velocity'] = P
            train_episodes, valid_episodes = sampler.sample(tasks,
                                                            num_steps=config['num-steps'],
                                                            fast_lr=config['fast-lr'],
                                                            gamma=config['gamma'],
                                                            gae_lambda=config['gae-lambda'],
                                                            device=args.device)

            logs['tasks'].extend(tasks)
            train_returns.append(get_returns(train_episodes[0]))
            valid_returns.append(get_returns(valid_episodes))

        logs['train_returns'] = np.concatenate(train_returns, axis=0)
        logs['valid_returns'] = np.concatenate(valid_returns, axis=0)


        Results = logs
        returns = Results['train_returns']
        if args.output is not None:
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                log_filename = os.path.join(args.output, 'logs.txt')

        result_logs={}
        result_logs["Velocity"] = P
        valid_returns = Results['valid_returns']

        # print(returns)

        mean_return = np.mean(returns, axis=1)
        mean_valid_return = np.mean(valid_returns, axis=1)

        results, valid_results = [], []
        for i in range(args.num_batches):
            valid_results.append(np.mean(mean_valid_return[i*20:(i+1)*20]))
            results.append(np.mean(mean_return[i*20:(i+1)*20]))

        valid_results, results = np.array(valid_results), np.array(results)

        print("Valid mean", np.mean(valid_results), "Valid var", np.var(valid_results))
        result_logs["Valid mean"] = np.mean(valid_results).tolist()
        result_logs["Valid var"] = np.var(valid_results).tolist()

        # plt.plot((valid_results), 'g^')
        # plt.title('valid results')
        # plt.show()

        print("Mean", np.mean(results), "Var", np.var(results))
        result_logs["Mean"] = np.mean(results).tolist()
        result_logs["Var"] = np.var(results).tolist()

        if args.output is not None:
                    with open(log_filename, 'a') as f:
                        json.dump(result_logs, f, indent=2)
                    f.close()
        point += 1

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
