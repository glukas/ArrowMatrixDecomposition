import numpy as np
import os
import random
import argparse

import matplotlib.pyplot as plt

from scipy import sparse
from tqdm import tqdm

print('Start...')

random.seed(0)
np.random.seed(0)


def print_stats(dataset_name: str, dataset_dir: str = None, save_dir: str = None):

    # here to make sure it runs, TODO: eventually remove for anonymity
    if dataset_dir is None:
        if dataset_name == "ogbn-papers100M":
            dataset_dir = f"/scratch/aziogas/datasets"
        else:
            dataset_dir = f"/scratch/sashkboo/datasets/{dataset_name}"
    if save_dir is None:
        save_dir = "/scratch/sashkboo/datasets/"

    A = sparse.load_npz(os.path.join(dataset_dir, f"{dataset_name}_A.npz"))

    nnode = A.shape[0]
    nedge = A.nnz

    out_degs = np.diff(A.indptr)

    At = A.transpose().tocsr()
    in_degs = np.diff(At.indptr)

    out_max_deg = np.max(out_degs)
    out_min_deg = np.min(out_degs)
    out_avg_deg = np.mean(out_degs)

    in_max_deg = np.max(in_degs)
    in_min_deg = np.min(in_degs)
    in_avg_deg = np.mean(in_degs)

    print(
        f"Dataset {dataset_name} has {nnode} nodes, {nedge} edges, out_max_deg={out_max_deg}, out_min_deg={out_min_deg}, out_avg_deg={out_avg_deg}, in_max_deg={in_max_deg}, in_min_deg={in_min_deg}, in_avg_deg={in_avg_deg}.")

    if not os.path.exists(os.path.join(save_dir, f"stats_{dataset_name}.txt")):
        with open(os.path.join(save_dir, f"stats_{dataset_name}.txt"), 'w') as f:
            f.write(
                f"Dataset {dataset_name} has {nnode} nodes, {nedge} edges, out_max_deg={out_max_deg}, out_min_deg={out_min_deg}, out_avg_deg={out_avg_deg}, in_max_deg={in_max_deg}, in_min_deg={in_min_deg}, in_avg_deg={in_avg_deg}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset_dir', type=str, default='./datasets')
    # parser.add_argument('--save_dir', type=str, default='./datasets')

    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('--dataset_names', type=list[str], nargs='+', default=['ogbn-papers100M'])
    # Datasets used in the project:
    # ['kmer_V2a', 'kmer_A2a', 'mawi_201512020130', 'mawi_201512012345', 'mawi_201512020000', 'mawi_201512020030',
    # 'mawi_201512020330', 'webbase-2001', 'ogbn-papers100M']

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset_names = args.dataset_names
    save_dir = args.save_dir

    for dataset_name in dataset_names:
        print_stats(dataset_name, dataset_dir, save_dir)

## TODO: remove, for testing, run with arguments --dataset_dir=None --save_dir=None
