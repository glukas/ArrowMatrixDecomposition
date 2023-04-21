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


def compute_cuthill_mckee_bandwidth(dataset_name: str, dataset_dir: str = None, save_dir: str = None):

    if not os.path.exists(os.path.join(save_dir, f"{dataset_name}_A_CM_perm.npz")):

        A = sparse.load_npz(os.path.join(dataset_dir, f"{dataset_name}_A.npz"))
        print("Computing permutation...")

        # compute permutations: different settings depending on the directedness of the graph
        if dataset_name in ["webbase-2001", "ogbn-papers100M"]:
            permutation = sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=False)
        else:
            permutation = sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
        print("Done computing permutation. Computing permuted csr matrix...")

        A_coo = A.tocoo(copy=True)
        perm_sort = np.asarray(np.argsort(permutation), dtype=A_coo.row.dtype)
        A_coo.row = perm_sort[A_coo.row]
        A_coo.col = perm_sort[A_coo.col]

        A_perm = A_coo.tocsr()

        sparse.save_npz(os.path.join(save_dir, f"{dataset_name}_A_CM_perm.npz"), A_perm)
    else:
        A_perm = sparse.load_npz(os.path.join(save_dir, f"{dataset_name}_A_CM_perm.npz"))

    # compute bandwidth
    abs1 = np.abs(A_perm.indices[np.maximum(0, A_perm.indptr[1:] - 1)] - range(len(A_perm.indptr) - 1))
    abs2 = np.abs(range(len(A_perm.indptr) - 1) - A_perm.indices[np.minimum(len(A_perm.indptr), A_perm.indptr[:-1])])

    bandwidth_vec = np.maximum(abs1, abs2)
    bandwidth = max(bandwidth_vec)

    if not os.path.exists(os.path.join(save_dir, f"bandwidth_{dataset_name}.txt")):
        with open(os.path.join(save_dir, f"bandwidth_{dataset_name}.txt"), 'w') as f:
            f.write(
                f"The bandwidth of the permuted matrix resulting from the RCM algorithm on dataset {dataset_name} has bandwidth {bandwidth}.")

    print(
        f"The bandwidth of the permuted matrix resulting from the RCM algorithm on dataset {dataset_name} has bandwidth {bandwidth}.")

    return bandwidth, A_perm


def visualize_cm_permutation(dataset_name: str, dataset_dir: str=None, save_dir: str = None):
    print(f"Visualizing CM permutation for {dataset_name}...")
    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    # here to make sure it runs, TODO: eventually remove for anonymity
    if dataset_dir is None:
        if dataset_name == "ogbn-papers100M":
            dataset_dir = f"/scratch/aziogas/datasets"
        else:
            dataset_dir = f"/scratch/sashkboo/datasets/{dataset_name}"
    if save_dir is None:
        save_dir = "/scratch/sashkboo/datasets/"

    A = sparse.load_npz(os.path.join(dataset_dir, f"{dataset_name}_A.npz"))
    A_perm = sparse.load_npz(os.path.join(save_dir, f"{dataset_name}_A_CM_perm.npz"))

    axs[0].spy(A, markersize=0.5)
    axs[0].set_title("A")

    axs[1].spy(A_perm, markersize=0.5)
    axs[1].set_title("PAP^T")

    plt.show()
    plt.savefig(os.path.join(save_dir, f"CM_permutation_{dataset_name}.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_dir', type=str, default='./datasets')
    # parser.add_argument('--save_dir', type=str, default="./datasets")

    parser.add_argument('--dataset_dir', type=str, default=None)  # TODO: eventually remove for anonymity
    parser.add_argument('--save_dir', type=str, default=None)  # TODO: eventually remove for anonymity

    parser.add_argument('--undirected_graph', type=bool, default=True)
    parser.add_argument('--dataset_names', type=list[str], nargs='+', default=['ogbn-papers100M'])
    # Datasets used in the project:
    # ['kmer_V2a', 'kmer_A2a', 'mawi_201512020130', 'mawi_201512012345', 'mawi_201512020000', 'mawi_201512020030',
    # 'mawi_201512020330', 'webbase-2001', 'ogbn-papers100M']

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset_names = args.dataset_names
    save_dir = args.save_dir

    for dataset_name in dataset_names:
        compute_cuthill_mckee_bandwidth(dataset_name, dataset_dir, save_dir)
        visualize_cm_permutation(dataset_name, dataset_dir, save_dir)

## TODO: remove, for testing, run with arguments --dataset_dir=None --save_dir=None
