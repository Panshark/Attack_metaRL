import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description="test drawer")
parser.add_argument('--resultpath', type=str, required=True,
        help='path to the npz file directory')
parser.add_argument('--output_folder', type=str, required=True,
        help='path to the logs file directory')
parser.add_argument('--num-batches', type=int, required=True, default=10,
        help= 'the number of batches during the test')
parser.add_argument('--num-traj', type=int, required=True, default=20,
        help='the number of trajectories per batch')
args = parser.parse_args()

Results = np.load(os.path.dirname(os.path.realpath(__file__)) + '/' + args.resultpath + '/results.npz', allow_pickle = True)
returns = Results['train_returns']
if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        log_filename = os.path.join(args.output_folder, 'logs.txt')

logs={}
logs["Name"] = args.resultpath
valid_returns = Results['valid_returns']

# print(returns)

mean_return = np.mean(returns, axis=1)
mean_valid_return = np.mean(valid_returns, axis=1)

results, valid_results = [], []
for i in range(args.num_batches):
    valid_results.append(np.mean(mean_valid_return[i*args.num_traj:(i+1)*args.num_traj]))
    results.append(np.mean(mean_return[i*args.num_traj:(i+1)*args.num_traj]))

valid_results, results = np.array(valid_results), np.array(results)

print("Valid mean", np.mean(valid_results), "Valid var", np.var(valid_results))
logs["Valid mean"] = np.mean(valid_results).tolist()
logs["Valid var"] = np.var(valid_results).tolist()

# plt.plot((valid_results), 'g^')
# plt.title('valid results')
# plt.show()

print("Mean", np.mean(results), "Var", np.var(results))
logs["Mean"] = np.mean(results).tolist()
logs["Var"] = np.var(results).tolist()

if args.output_folder is not None:
            with open(log_filename, 'a') as f:
                json.dump(logs, f, indent=2)
            f.close()

# plt.plot((results), 'r^')
# plt.title('fast adaptation train results')
# plt.show()     