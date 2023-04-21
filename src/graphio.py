import igraph
import numpy as np
import pickle

from preprocessing.arrowdecomposition import ArrowGraph
from numpy import typing as npt
from scipy import sparse
from typing import List, Union


def load_graph(filename: str) -> igraph.Graph:
    """
    :param filename: 
    :return: 
    """
    return pickle.load(open(f"{filename}_graph.pickle", "rb"))


def save_decomposition(graph: igraph.Graph,
                       decomposition: list[ArrowGraph],
                       filename: str,
                       dtype: npt.DTypeLike = np.float32,
                       use_width: bool = True,
                       block_diagonal: bool = True,
                       saveGraph: bool = True) -> None:
    """
    Saves the decomposition to files in scipy csr format
    The i-th part of the decomposition is stored as {filename}_B_{width}_{bd}_{i}.npz
    The permutation that maps the original id's to the id's of B_i is in {filename}_B_{width}_{bd}_{i}_permutation.py
    as a numpy array.
    :param decomposition: the decomposition to store
    :param filename: prefix to use for storing the files
    :param dtype: the data type to use for the sparse matrices
    :param use_width: ignored. exists for backwards compatibility.
    :param block_diagonal: whether the decomposition uses the block diagonal in the filename
    :return: None
    """

    if saveGraph:
        # Save graph
        pickle.dump(graph, open(f"{filename}_graph.pickle", "wb"))

        # Save A
        A = graph.get_adjacency_sparse().astype(dtype)
        sparse.save_npz(f"{filename}_A.npz", A)

    # Save B
    for i, arrow in enumerate(decomposition):
        A = arrow.graph.get_adjacency_sparse().astype(dtype)

        basename = get_pathname(filename, arrow.arrow_width, block_diagonal)

        sparse.save_npz(f"{basename}_{i}.npz", A)
        np.save(f"{basename}_{i}_permutation.npy", arrow.permutation)

    # Save nonzeros (for convenience)
    nonzero_rows = np.asarray([a.nonzero_rows for a in decomposition], dtype=np.int64)
    np.save(f"{basename}_nonzeros.npy", nonzero_rows)


def load_decomposition(filename: str, width: int = None, block_diagonal: bool = True, no_permutation=False) \
        -> list[(sparse.csr_matrix, Union[None, npt.NDArray[np.integer]])]:
    """
    Loads the decomposition from files in scipy csr format
    The i-th part of the decomposition is stored as {filename}_B_{width}_{bd}_{i}.npz
    The permutation that maps the original id's to the id's of B_i is in {filename}_B_{width}_{bd}_{i}_permutation.py
    as a numpy array.
    :param no_permutation: If true, the permutation matrix is not loaded (None is at its place)
    :param filename: prefix to use for loading the files
    :param width: The width of the arrow to load. If None, it is assume that it is not part of the filename.
    :param block_diagonal: whether the decomposition uses the block diagonal format
    :return: the decomposition
    """

    # Load B
    i = 0
    decomposition = []

    basename = get_pathname(filename, width, block_diagonal)
    print("Loading decomposition", basename, "...")
    while True:

        try:
            B = sparse.load_npz(f"{basename}_{i}.npz")
            if no_permutation:
                permutation = None
            else:
                permutation = np.load(f"{basename}_{i}_permutation.npy")
        except FileNotFoundError:
            break
        decomposition.append((B, permutation))
        i += 1

    if len(decomposition) == 0:
        # THIS IS THE OLD NAMING SCHEME.
        # TO SUPPORT THE OLD NAMING SCHEME, WE SEARCH FOR IT ALSO IF THE PREVIOUS BREAKS
        while True:
            try:
                #mawi_201512020130_B_5000000_0_bd
                #mawi_201512020130_B_5000000_0_bd_permutation.npy
                basename = f"{filename}_B"
                if width:
                    basename += f"_{width}"
                basename += f"_{i}"
                if block_diagonal:
                    basename += "_bd"
                B = sparse.load_npz(f"{basename}.npz")
                print("matrix found:", B.nnz)
                if no_permutation:
                    permutation = None
                else:
                    permutation = np.load(f"{basename}_permutation.npy")
            except FileNotFoundError:
                break
            decomposition.append((B, permutation))
            i += 1

    return decomposition


def split_matrix_to_blocks(A: sparse.csr_matrix,
                           block_size: int,
                           dtype: npt.DTypeLike = None,
                           use_min_shape: bool = False) -> List[List[Union[sparse.csr_matrix, None]]]:
    """
    Splits the matrix A into blocks of size block_size x block_size
    :param A: the matrix to split
    :param block_size: the size of the blocks
    :param dtype: the data type to use for the blocks. If None, the data type of A is used.
    :param use_min_shape: whether to use the minimum shape of the blocks or keep it fixed at block_size
    :return: a list of the blocks
    """
    rows, cols = A.shape
    dtype = dtype or A.dtype

    # Generate blocks
    blocks_per_col = int(np.ceil(rows / block_size))
    blocks_per_row = int(np.ceil(cols / block_size))
    blocks = [[None for _ in range(blocks_per_row)] for _ in range(blocks_per_col)]
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            if i > 0 and not j in (0, i - 1, i, i + 1):
                continue
            #

            shape = (min(rows - i * block_size, block_size), min(cols - j * block_size, block_size))
            slice = A[i * block_size:min(rows, (i + 1) * block_size),
                                        j * block_size:min(cols, (j + 1) * block_size)]
            pad_width = block_size - shape[0]

            if use_min_shape or pad_width == 0:
                block = sparse.csr_matrix(slice, shape=shape, dtype=dtype)
            else:
                # We need to pad the index pointer so that there are enough rows
                shape2 = (block_size, block_size)
                indx_ptr = np.pad(slice.indptr, (0, pad_width), mode='edge')
                block = sparse.csr_matrix((slice.data, slice.indices, indx_ptr),
                                          shape=shape2,
                                          dtype=dtype)

            block.sum_duplicates()
            block.sort_indices()
            assert block.has_canonical_format
            blocks[i][j] = block
    
    return blocks


def get_pathname(basename: str, width: int, is_block_diagonal: bool):
    basename = f"{basename}_B"
    if width:
        basename += f"_{width}"
    if is_block_diagonal:
        basename += "_bd"
    return basename
