import argparse
import igraph
import numpy as np
import os
import pickle
import random

import graphio
from arrowdecomposition import arrow_decomposition, ArrowGraph
from numpy import typing as npt
from scipy import sparse
from typing import Any, Dict, Tuple
import scipy
import mat73
from tqdm import tqdm


random.seed(0)
np.random.seed(0)

def load_graph_matlab(filename: str, undirected: bool=False) -> igraph.Graph:
    """
    :param filename: filename of the matlab matrix file
    :param undirected: whether the graph is undirected
    :return: the graph as an igraph object
    """
    # dataset = scipy.io.loadmat(filename)
    dataset = mat73.loadmat(filename)
    # mat = dataset['Problem'][0][0][2]
    mat = dataset['Problem']['A']
    mode = "undirected" if undirected else "directed"
    mat = mat.tocoo()
    edgelist = np.stack((mat.row, mat.col), axis=1)
    return  igraph.Graph.TupleList(edgelist, directed=not undirected)

if __name__ == "__main__":
    # Download the datasets

    parser = argparse.ArgumentParser()

    parser.add_argument('--width', type=int, default=5000000)
    parser.add_argument('--dataset_dir', type=str, default='../../datasets/custom')
    parser.add_argument('--dataset_names', type=list[str], nargs='+', default=['kmer_V2a', 'kmer_A2a', 'mawi_201512020130', 'mawi_201512012345', 'mawi_201512020000', 'mawi_201512020030', 'mawi_201512020330', 'webbase-2001', 'ogbn-papers100M'])

    args = parser.parse_args()

    datasets_directory = args['dataset_dir']
    dataset_names = args['dataset_names']
    width = args['width']

    block_diagonal = True

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(datasets_directory, dataset_name)
        dataset_mat_file = dataset_name + '.mat'
        decomposition_dir = os.path.join(datasets_directory, dataset_name)

        print(f"Loading {dataset_name}'s graph...")
        graph = pickle.load(open(os.path.join(dataset_dir, f"{dataset_name}_graph.pickle"), "rb"))

        print(f"Converting {dataset_name} to arrow decomposition with width {width}...")
        B = arrow_decomposition(graph, arrow_width=width, max_number_of_levels=5, block_diagonal=block_diagonal)

        for i, arrow in tqdm(enumerate(B)):
            print(f"Converting the {i}-th arrow matrix to sparse matrix...")
            matrix = arrow.graph.get_adjacency_sparse().astype(np.float32)
            basename = f"{dataset_name}_B_{arrow.arrow_width}_{i}"
            if block_diagonal:
                basename = f"{basename}_bd"
            sparse.save_npz(os.path.join(dataset_dir, f"{basename}.npz"), matrix)
            np.save(os.path.join(dataset_dir, f"{basename}_permutation.npy"), arrow.permutation)
            
        
        