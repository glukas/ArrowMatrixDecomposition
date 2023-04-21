import argparse
import time

import cupy as cp
import numpy as np
import os
import scipy as sp

import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat
from typing import Dict, List, Tuple, Union

import wb_logging


def _sp2cp(matrix: sparse.csr_matrix) -> cp.sparse.csr_matrix:
    """ Converts a SciPy CSR matrix to a CuPy CSR matrix. 
    
    :param matrix: The SciPy CSR matrix.
    :return: The CuPy CSR matrix.
    """
    tmp = cp.sparse.csr_matrix((cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
                               shape=matrix.shape,
                               dtype=matrix.dtype)
    tmp._has_canonical_format = True
    return tmp


def generate_2D_decomposition(grid: Dict[int, Tuple[int, int]], A: sparse.csr_matrix, X_cols: int, dtype, rng: np.random.Generator):

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    if world_size not in grid:
        utils.mpi_print(world_rank, f"Invalid grid size {world_size}.")
        exit(1)  

    # Cartesian grid
    Px, Py = grid[world_size]
    cart_comm = world_comm.Create_cart((Px, Py))
    cart_rank = cart_comm.Get_rank()
    x, y = cart_comm.Get_coords(cart_rank)

    assert world_rank == cart_rank

    # Subcommunicators
    bcast_comm = cart_comm.Sub([True, False])
    bcast_rank = bcast_comm.Get_rank()
    reduce_comm = cart_comm.Sub([False, True])

    # Global sizes
    utils.mpi_print(cart_rank, "Broadcasting global sizes...")
    if cart_rank == 0:
        global_sizes = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64)
    else:
        global_sizes = np.empty(3, dtype=np.int64)
    cart_comm.Bcast(global_sizes, root=0)
    NI, NK, NNZ = global_sizes
    NJ = X_cols

    # Local sizes
    lNI, lNK = int(np.ceil(NI / Px)), int(np.ceil(NK / Py))
    lNJ = NJ
    lNNZ = int(np.ceil(NNZ / world_size))

    # Distribute the adjacency matrix
    utils.mpi_print(cart_rank, "Distributing the adjacency matrix...")
    lA = None
    if cart_rank == 0:
        for i in range(Px):
            for j in range(Py):
                block = sparse.csr_matrix(A[i * lNI:min(NI, (i + 1) * lNI), j * lNK:min(NK, (j + 1) * lNK)])
                block.sum_duplicates()
                block.sort_indices()
                block._has_canonical_format = True
                if x == i and y == j:
                    lA = block
                    lNI = block.shape[0]
                    lNK = block.shape[1]
                    lNNZ = block.nnz
                else:
                    dst = cart_comm.Get_cart_rank((i, j))
                    size_buffer = np.array([block.shape[0], block.shape[1], block.nnz], dtype=np.int32)
                    cart_comm.Send(size_buffer, dest=dst, tag=0)
                    cart_comm.Send(block.indptr, dest=dst, tag=1)
                    cart_comm.Send(block.indices, dest=dst, tag=2)
                    cart_comm.Send(block.data, dest=dst, tag=3)
    else:
        size_buffer = np.empty(3, dtype=np.int32)
        cart_comm.Recv(size_buffer, source=0, tag=0)
        lNI, lNK, lNNZ = size_buffer
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=dtype)
        cart_comm.Recv(indptr, source=0, tag=1)
        cart_comm.Recv(indices, source=0, tag=2)
        cart_comm.Recv(data, source=0, tag=3)
        lA = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=dtype)

    cart_comm.Barrier()

    # The X matrix is replicated in the "bcast" communicators.
    # Therefore, we generate a random block in bcast-rank 0 and then bcast.
    # TODO Measure cost of distributing X
    utils.mpi_print(cart_rank, f"Generating matrix X with shape ({NK}, {NJ})...")
    if bcast_rank == 0:
        X = utils.generate_dense_matrix(lNK, lNJ, dtype, rng)
    else:
        X = np.empty((lNK, lNJ), dtype=dtype)
    for i in range(0, lNK, 2**31):
        bcast_comm.Bcast(X[i:min(lNK, i + 2**31), :], root=0)
    
    Y = np.empty((lNI, lNJ), dtype=dtype)

    return lA, X, Y, cart_comm, bcast_comm, reduce_comm


def SpMM2D_cpu(A: sparse.csr_matrix, X: np.ndarray, Y: np.ndarray, cart_comm: MPI.Cartcomm, bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm):
    """ Performs the SpMM computation on a 2D grid. CPU execution. """
    tic = time.perf_counter()
    Y[:] = A @ X
    toc = time.perf_counter()
    wb_logging.log({"spmm_kernel_time": toc-tic})
    tic = time.perf_counter()
    reduce_comm.Allreduce(MPI.IN_PLACE, Y, op=MPI.SUM)
    toc = time.perf_counter()
    tic = time.perf_counter()
    wb_logging.log({"spmm_reduce_time": toc-tic})
    bcast_comm.Bcast(X, 0)
    toc = time.perf_counter()
    wb_logging.log({"spmm_bcast_time": toc-tic})
    return Y


def SpMM2D_gpu(A: sparse.csr_matrix, X: np.ndarray, Y: np.ndarray, cart_comm: MPI.Cartcomm, bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm):
    """ Performs the SpMM computation on a 2D grid. GPU execution. """
    tic = time.perf_counter()
    Y[:] = cp.asnumpy(_sp2cp(A) @ cp.asarray(X))
    toc = time.perf_counter()
    wb_logging.log({"spmm_kernel_time": toc-tic})
    tic = time.perf_counter()
    reduce_comm.Allreduce(MPI.IN_PLACE, Y, op=MPI.SUM)
    toc = time.perf_counter()
    wb_logging.log({"spmm_reduce_time": toc-tic})
    tic = time.perf_counter()
    bcast_comm.Bcast(X, 0)
    toc = time.perf_counter()
    wb_logging.log({"spmm_bcast_time": toc-tic})
    return Y


grid = {
    #     [Px, Py]
    1: [1, 1],
    2: [1, 2],
    4: [2, 2],
    8: [2, 4],
    16: [4, 4],
    32: [4, 8],
    64: [8, 8],
    128: [8, 16],
    256: [16, 16],
    512: [16, 32],
    1024: [32, 32]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SpMM2D benchmark.')
    parser.add_argument('-d',
                        '--dataset',
                        nargs="?",
                        choices=['random', 'file'],
                        default='random',
                        help='The source of the sparse matrix.')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        nargs="?",
                        default=42,
                        help='The seed for the random number generator.')
    parser.add_argument('-v',
                        '--vertices',
                        type=int,
                        nargs="?",
                        default=100000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=1000000, help='The number of edges in the graph.')
    parser.add_argument('-t',
                        '--type',
                        nargs="?",
                        choices=['float32', 'float64'],
                        default='float32',
                        help='The type of the data.')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing the sparse matrix.')
    parser.add_argument('-c', '--columns', type=int, nargs="?", default=128, help='The number of columns in the matrix X.')
    parser.add_argument('--validate', type=utils.str2bool, nargs="?", default=False, help='Validate the result.')
    parser.add_argument('-k', '--wandb_key', type=str, default=None, help='Wandb API key.')
    parser.add_argument('-i', '--device', type=str, default='gpu', help="Either gpu or cpu.")
    parser.add_argument('-z', '--iterations', type=int, default=10, help="Number of iterations to benchmark.")

    args = vars(parser.parse_args())

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    if world_size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")

    # A matrix
    A = None
    if args['dataset'] == 'file':
        if args['file'] is None:
            utils.mpi_print(world_rank, "Please specify the file contaning the adjacency matrix.")
            exit(1)
        absolute_path = os.path.abspath(args['file'])
        if not os.path.exists(absolute_path):
            utils.mpi_print(world_rank, f"The file {args['file']} does not exist.")
            exit(1)
        folder, filename = os.path.split(absolute_path)
        if not filename.endswith('.npz'):
            utils.mpi_print(world_rank, f"The file {args['file']} is not a .npz file.")
            exit(1)
        utils.mpi_print(world_rank, f"Loading adjacency matrix from {args['file']}...")
        if world_rank == 0:
            A = sparse.load_npz(absolute_path)
            if A.dtype != dtype:
                utils.mpi_print(world_rank, f"Converting matrix from {A.dtype} to {dtype}...")
                A = A.astype(dtype)
    elif args['dataset'] == 'random':
        utils.mpi_print(
            world_rank,
            f"Generating random adjacency matrix with {args['vertices']} vertices and {args['edges']} edges...")
        if world_rank == 0:
            A = utils.generate_sparse_matrix(args['vertices'], args['vertices'], args['edges'], dtype, rng)
    else:
        raise NotImplementedError
    
    lA, lX, lY, cart_comm, bcast_comm, reduce_comm = generate_2D_decomposition(grid, A, args['columns'], dtype, rng)

    # Validation
    if args['validate']:
        utils.mpi_print(world_rank, "Validating the result...")
        Px, Py = grid[world_size]
        bcast_rank = bcast_comm.Get_rank()
        reduce_rank = reduce_comm.Get_rank()

        world_comm.Barrier()
        print(f"world_rank: {world_rank}, bcast_rank: {bcast_rank}, reduce_rank: {reduce_rank}", flush=True)
        world_comm.Barrier()


        # Gather X
        if world_rank == 0:
            X = np.empty((A.shape[1], args['columns']), dtype)
            X[:lX.shape[0]] = lX
            idx = lX.shape[0]
            for r in range(1, Py):
                start = idx
                end = min(X.shape[0], idx + lX.shape[0])
                world_comm.Recv(X[start:end], source=r)
        elif world_rank < Py:
            assert bcast_rank == 0
            world_comm.Send(lX, dest=0)

        world_comm.Barrier()

        if world_rank == 0:
            ref = A @ X
        
        SpMM2D_cpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)

        # Gather Y
        if world_rank == 0:
            Y = np.empty((A.shape[0], args['columns']), dtype)
            Y[:lY.shape[0]] = lY
            idx = lY.shape[0]
            for r in range(Py, world_size, Py):
                start = idx
                end = min(Y.shape[0], idx + lY.shape[0])
                world_comm.Recv(Y[start:end], source=r)
            utils.mpi_print(world_rank, f"CPU validation: {np.allclose(Y, ref)}")
        elif world_rank % Py == 0:
            assert reduce_rank == 0
            world_comm.Send(lY, dest=0)
        
        world_comm.Barrier()

        SpMM2D_gpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)

        # Gather Y
        if world_rank == 0:
            Y = np.empty((A.shape[0], args['columns']), dtype)
            Y[:lY.shape[0]] = lY
            idx = lY.shape[0]
            for r in range(Py, world_size, Py):
                start = idx
                end = min(Y.shape[0], idx + lY.shape[0])
                world_comm.Recv(Y[start:end], source=r)
            utils.mpi_print(world_rank, f"GPU validation: {np.allclose(Y, ref)}")
        elif world_rank % Py == 0:
            assert reduce_rank == 0
            world_comm.Send(lY, dest=0)

    n_iterations = args['iterations']

    dataset_name = 'random'
    if args['dataset'] == 'file':
        dataset_name = args['file']

    if args['device'] == 'cpu':
        wb_logging.wandb_init(world_comm, dataset_name, args['columns'], n_iterations, 'cpu', '2DAlex_v0.2', lX.shape[0], args['wandb_key'])

        # Benchmark
        utils.mpi_print(world_rank, "Benchmarking on CPU...")
        cpu_runtimes = repeat("SpMM2D_cpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)",
                              setup="cart_comm.Barrier()",
                              repeat=n_iterations,
                              number=1,
                              globals={**locals(), **globals()})

        for i, sample in enumerate(cpu_runtimes):
            wb_logging.log({'spmm_time': sample, 'iteration': i})

        utils.mpi_print(world_rank,
                        f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
        wb_logging.finish()

    else:

        wb_logging.wandb_init(world_comm, dataset_name, args['columns'], n_iterations, 'gpu', '2DAlex_v0.2', lX.shape[0], args['wandb_key'])

        utils.mpi_print(world_rank, "Benchmarking on GPU...")
        gpu_stmt = "SpMM2D_gpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)"
        gpu_setup = "cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
        gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=n_iterations, number=1, globals={**locals(), **globals()})
        utils.mpi_print(world_rank,
                        f"GPU: {utils.time_to_ms(np.median(gpu_runtimes))} +- {utils.time_to_ms(np.std(gpu_runtimes))}")

        for i, sample in enumerate(gpu_runtimes):
            wb_logging.log({'spmm_time': sample, 'iteration': i})

        wb_logging.finish()
