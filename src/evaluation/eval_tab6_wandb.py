WANDB_PROJECT = "spmm-mpi"

# HERE, YOU NEED TO PUT YOUR API KEY, IF NEEDED. OTHERWISE, IT ONLY WORKS FOR ANONYMOUS PROJECTS
WANDB_API_KEY = None

import wandb
import numpy as np

def spmm_time_calc_stat(spmm_times_list, iterations, ranks):
    assert iterations * ranks == len(spmm_times_list)

    times = np.zeros(iterations - 1)
    for rank in range(ranks):
        for i in range(iterations):
            if i != 0:
                times[i - 1] = max(times[i - 1], spmm_times_list[(rank) * iterations + i])

    mean = np.mean(times)
    min_res = np.min(times)
    max_res = np.max(times)
    std_res = np.std(times)

    return mean, min_res, max_res, std_res


def map_mawi_dataset(mawi_dataset_name):
    mawi_mapper = {
        'mawi_201512012345': '19M',
        'mawi_201512020000': '36M',
        'mawi_201512020030': '69M',
        'mawi_201512020130': '129M',
        'mawi_201512020330': '226M'
    }
    return mawi_mapper[mawi_dataset_name[:17]]


def get_runs(algorithm_name):
    api = wandb.Api(api_key=WANDB_API_KEY)
    runs = api.runs(WANDB_PROJECT)

    proper_runs = []
    for run in runs:

        if 'device' not in run.config or run.config['device'] == 'cpu':
            continue
        if 'algorithm' not in run.config or \
                run.config['algorithm'] != algorithm_name:  # not in ['2DAlex_v0.2', 'Arrow_v0.2_BlockDiagonal']:
            continue
        if 'dataset' not in run.config or 'mawi' not in run.config['dataset']:
            continue
        if 'n_features' not in run.config:
            continue
        if run.config['algorithm'] == 'Arrow_v0.2_BlockDiagonal' and run.config['width'] != 5000000:
            continue
        if run.state == 'failed':
            continue

        rank_num = run.config['ranks']
        if 'actual_ranks' in run.summary.keys():
            rank_num = run.summary['actual_ranks']
        iter_num = run.config['iterations']
        algorithm = run.config['algorithm']
        state = run.state
        n_features = run.config['n_features']

        if iter_num <= 3:
            continue
        history = run.scan_history()
        spmm_times = []
        for row in history:
            if 'spmm_time' in row:
                spmm_times.append(row['spmm_time'])
        if len(spmm_times) == 0:
            continue

        if iter_num * rank_num != len(spmm_times):
            continue

        time_mean, time_min, time_max, time_std = spmm_time_calc_stat(spmm_times, iter_num, rank_num)
        if time_max - time_min > 0.05 * time_mean:
            print(
                f'dataset: {run.config["dataset"]}, rank_num: {rank_num}, n_features: {n_features}, mean: {time_mean}, min: {time_min}, max: {time_max}, std: {time_std}')

        raw_data_dic = {
            'dataset': map_mawi_dataset(run.config['dataset']),
            'rank_num': rank_num,
            'iter_num': iter_num,
            'algorithm': algorithm,
            'state': state,
            'spmm_time': time_mean,
            'spmm_time_min': time_min,
            'spmm_time_max': time_max,
            'n_features': n_features
        }

        proper_runs.append(raw_data_dic)
    return proper_runs


runs_arrow = get_runs('Arrow_v0.2_BlockDiagonal')
runs_baseline = get_runs('2DAlex_v0.2')

print(runs_arrow)
print(runs_baseline)