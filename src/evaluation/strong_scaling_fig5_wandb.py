import numpy as np
import matplotlib.pyplot as plt
import wandb

WANDB_PROJECT = "spmm-mpi"

#HERE, YOU NEED TO PUT YOUR API KEY, IF NEEDED. OTHERWISE, IT ONLY WORKS FOR ANONYMOUS PROJECTS
WANDB_API_KEY = None

def spmm_time_calc(spmm_times_list, iterations, ranks):
    assert iterations*ranks == len(spmm_times_list)
    valid_results = []
    for rank in range(ranks):
        for i in range(iterations):
            if i != 0:
                valid_results.append(spmm_times_list[(rank)*iterations + i])
    return np.max(valid_results)
                
def map_mawi_dataset(mawi_dataset_name):
    mawi_mapper = {
        'mawi_201512012345': '19M',
        'mawi_201512020000': '36M',
        'mawi_201512020030': '69M',
        'mawi_201512020130': '129M',
        'mawi_201512020330': '226M',
        'mawi_201512020030_A.npz': '69M',
    }
    return mawi_mapper[mawi_dataset_name]    


def get_runs(dataset_name):

    api = wandb.Api(api_key=WANDB_API_KEY)

    runs = api.runs(WANDB_PROJECT)

    print(runs)

    proper_runs = []
    for run in runs: 

        if 'device' not in run.config or run.config['device'] == 'cpu': 
            continue
        if 'dataset' not in run.config or run.config['dataset'] != dataset_name:
            continue
        if 'n_features' not in run.config:
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

        spmm_time = spmm_time_calc(spmm_times, iter_num, rank_num)
        
        raw_data_dic = {
            'dataset': map_mawi_dataset(run.config['dataset']),
            'rank_num': rank_num,
            'iter_num': iter_num,
            'algorithm': algorithm,
            'state': state,
            'spmm_time':spmm_time,
            'n_features': n_features
        }
        if raw_data_dic not in proper_runs:
            proper_runs.append(raw_data_dic)
    return proper_runs



runs_arrow = get_runs("mawi_201512020030")
runs_baseline = get_runs("mawi_201512020030_A.npz")


n_features_list = [128, 64, 32]
colors = ['blue', 'green', 'red']
markers = ['o', 's', 'd']
fig, ax = plt.subplots(ncols=1, figsize=(6, 4))
ax.grid()
ax.set_axisbelow(True)
plt.axhspan(75, 110, facecolor="0.8")
for i, n_features in enumerate(n_features_list):
    
    rank_num = []
    spmm_times = []
    for d in runs_arrow:
        if d['n_features'] == n_features:
            ranks = d['rank_num']
            if ranks not in rank_num:
                rank_num.append(ranks)
                spmm_times.append(d['spmm_time'])

    spmm_times = [x for x, _ in sorted(zip(spmm_times, rank_num), key=lambda pair: pair[1])]
    plt.plot(sorted(rank_num), spmm_times, 'o-', label=f'Arrow: K={n_features}', color=colors[i])


    rank_num = []
    spmm_times = []
    if n_features == 128:
        rank_num += [16, 32, 64]
        spmm_times += [85, 85, 85]
        marker = '*'
    else:
        marker = '*--'
    for d in runs_baseline:
        if d['n_features'] == n_features:
            ranks = d['rank_num']
            if ranks not in rank_num:
                rank_num.append(ranks)
                spmm_times.append(d['spmm_time'])

    spmm_times = [x for x, _ in sorted(zip(spmm_times, rank_num), key=lambda pair: pair[1])]
    plt.plot(sorted(rank_num), spmm_times, marker, label=f"1.5D: K={n_features}" , color=colors[i])

plt.plot([16], [80], marker='*', color=colors[1])


plt.yticks(range(10, 80, 10))
plt.ylim(0, 90)
plt.text(90, 80, 'Out of Memory', fontsize=13, weight='bold')
ax.set_ylabel('Runtime (s)', fontsize='medium')
ax.set_xlabel('Number of GPUs', fontsize='medium')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, 
         )

fig.savefig('strong_scaling_MAWI.pdf', bbox_inches='tight')


