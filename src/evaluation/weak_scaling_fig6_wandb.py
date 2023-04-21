import numpy as np
import matplotlib.pyplot as plt
import wandb

WANDB_PROJECT = "spmm-mpi"

# HERE, YOU NEED TO PUT YOUR API KEY, IF NEEDED. OTHERWISE, IT ONLY WORKS FOR ANONYMOUS PROJECTS
WANDB_API_KEY = None

def spmm_time_calc(spmm_times_list, iterations, ranks):
    assert iterations*ranks == len(spmm_times_list)
    valid_results = []
    for rank in range(ranks):
        for i in range(iterations):
            if i != 0:
                valid_results.append(spmm_times_list[(rank)*iterations + i])
    return np.max(valid_results)

def spmm_time_calc_stat(spmm_times_list, iterations, ranks):
    assert iterations*ranks == len(spmm_times_list)

    times = np.zeros(iterations-1)
    for rank in range(ranks):
        for i in range(iterations):
            if i != 0:
                times[i-1] = max(times[i-1], spmm_times_list[(rank)*iterations + i])

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
    api = wandb.Api(timeout=30, api_key=WANDB_API_KEY)
    runs = api.runs(WANDB_PROJECT)

    proper_runs = []
    for run in runs: 

        if 'device' not in run.config or run.config['device'] == 'cpu':
            continue
        if 'algorithm' not in run.config or \
        run.config['algorithm'] != algorithm_name: #not in ['2DAlex_v0.2', 'Arrow_v0.2_BlockDiagonal']:
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
        
        if iter_num <=3:
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
            print(f'dataset: {run.config["dataset"]}, rank_num: {rank_num}, n_features: {n_features}, mean: {time_mean}, min: {time_min}, max: {time_max}, std: {time_std}')
        
        raw_data_dic = {
            'dataset': map_mawi_dataset(run.config['dataset']),
            'rank_num': rank_num,
            'iter_num': iter_num,
            'algorithm': algorithm,
            'state': state,
            'spmm_time':time_mean,
            'spmm_time_min':time_min,
            'spmm_time_max':time_max,
            'n_features': n_features
        }

        if algorithm == '2DAlex_v0.2' and 'mawi_201512020030' in run.config['dataset'] and rank_num !=32 :
            continue
        if algorithm == '2DAlex_v0.2' and 'mawi_201512020000' in run.config['dataset'] and rank_num != 16:
            continue
        if algorithm == '2DAlex_v0.2' and 'mawi_201512020130' in run.config['dataset'] and rank_num != 64:
            continue 
        if algorithm == '2DAlex_v0.2' and 'mawi_201512020330' in run.config['dataset'] and rank_num != 128:
            continue
        if algorithm == '2DAlex_v0.2' and 'mawi_201512012345' in run.config['dataset'] and rank_num != 8:
            continue

        if algorithm == 'Arrow_v0.2_BlockDiagonal' and 'mawi_201512020030' in run.config['dataset'] and rank_num !=27:
            continue
        if algorithm == 'Arrow_v0.2_BlockDiagonal' and 'mawi_201512020000' in run.config['dataset'] and run.config['ranks'] != 16:
            continue
        if algorithm == 'Arrow_v0.2_BlockDiagonal' and 'mawi_201512020130' in run.config['dataset'] and run.config['ranks'] != 64:
            continue 
        if algorithm == 'Arrow_v0.2_BlockDiagonal' and 'mawi_201512020330' in run.config['dataset'] and run.config['ranks'] != 128:
            continue
        if algorithm == 'Arrow_v0.2_BlockDiagonal' and 'mawi_201512012345' in run.config['dataset'] and run.config['ranks'] != 8:
            continue

        proper_runs.append(raw_data_dic)
    return proper_runs


runs_arrow = get_runs('Arrow_v0.2_BlockDiagonal')
runs_baseline = get_runs('2DAlex_v0.2')


n_features_list = [128, 64, 32]
colors = ['blue', 'green', 'red']
fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
axs[0].grid()
axs[1].grid()
axs[0].set_axisbelow(True)
axs[1].set_axisbelow(True)
axs[0].axhspan(100, 120, facecolor="0.8")
axs[1].axhspan(100, 120, facecolor="0.8")


for i, n_features in enumerate(n_features_list):
    
    dataset_num = []
    spmm_times = []
    spmm_times_min = []
    spmm_times_max = []
    for d in runs_arrow:
        if d['n_features'] == n_features:
            dataset_node_num = int(d['dataset'][:-1])
            if dataset_node_num not in dataset_num:
                dataset_num.append(dataset_node_num)
                spmm_times.append(d['spmm_time'])
                spmm_times_min.append(d['spmm_time_min'])
                spmm_times_max.append(d['spmm_time_max'])

    sorted_ind = np.argsort(np.asarray(dataset_num))
    dataset_num = np.asarray(dataset_num)[sorted_ind]
    spmm_times = np.asarray(spmm_times)[sorted_ind]
    spmm_times_min = np.asarray(spmm_times_min)[sorted_ind]
    spmm_times_max = np.asarray(spmm_times_max)[sorted_ind]
    spmm_times_error = np.stack((spmm_times - spmm_times_min, spmm_times_max - spmm_times), axis=0)
    axs[0].errorbar(dataset_num, spmm_times, yerr=spmm_times_error, fmt='o-', label=f'K={n_features}', color=colors[i])

    dataset_num = []
    spmm_times = []
    spmm_times_min = []
    spmm_times_max = []
    for d in runs_baseline:
        if d['n_features'] == n_features:
            dataset_node_num = int(d['dataset'][:-1])
            if dataset_node_num not in dataset_num:
                dataset_num.append(dataset_node_num)
                spmm_times.append(d['spmm_time'])
                spmm_times_min.append(d['spmm_time_min'])
                spmm_times_max.append(d['spmm_time_max'])

    sorted_ind = np.argsort(np.asarray(dataset_num))
    dataset_num = np.asarray(dataset_num)[sorted_ind]
    spmm_times = np.asarray(spmm_times)[sorted_ind]
    spmm_times_min = np.asarray(spmm_times_min)[sorted_ind]
    spmm_times_max = np.asarray(spmm_times_max)[sorted_ind]
    spmm_times_error = np.stack((spmm_times - spmm_times_min, spmm_times_max - spmm_times), axis=0)

    axs[1].errorbar(dataset_num, spmm_times, yerr=spmm_times_error, fmt='*--', label=f'K={n_features}', color=colors[i])

dataset_num = [36, 69, 129, 226]
spmm_times = [113, 113, 113, 113]

plt.plot(dataset_num, spmm_times, '*', color=colors[0], axes=axs[1])
plt.plot([226],[106], '*', color=colors[1], axes=axs[1])


axs[0].text(15, 107, 'Out of Memory', fontsize=14, weight='bold')

font_size = 11
x_dist = 170
axs[0].text(x_dist, 60, 'K=128', fontsize=font_size)
axs[0].text(x_dist, 30, 'K=64', fontsize=font_size)
axs[0].text(x_dist, 15, 'K=32', fontsize=font_size)


fontsize = 14
title_fontsize = 15
axs[0].set_ylabel('Runtime (s)', fontsize=fontsize)
axs[0].set_xlabel('#Vertices in millions', fontsize=fontsize)

axs[0].set_yticks(range(10, 100, 10))
axs[0].set_ylim(0, 120)
axs[0].tick_params(axis='both', labelsize=fontsize)
axs[0].set_title('Ours (Arrow)', fontdict={'fontsize': title_fontsize, 'fontweight': 'bold'})

axs[1].set_xlabel('#Vertices in millions', fontsize=fontsize)

axs[1].set_yticks(range(10, 100, 10), ["" for i in range(10, 100, 10)])
axs[1].tick_params(axis='y', length=0)
axs[0].tick_params(axis='x', labelsize=fontsize)
axs[1].set_ylim(0, 120)
axs[1].set_title('Baseline (1.5D)', fontdict={'fontsize': title_fontsize, 'fontweight': 'bold'})
fig.tight_layout()#pad=1.0
fig.savefig('weak_scaling_MAWI.pdf', bbox_inches='tight')

# In[ ]:




