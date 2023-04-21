import argparse
import sys
import time
from typing import Union
import igraph
import numpy as np
from mpi4py import MPI
from preprocessing import arrowdecomposition
import graphio
import wb_logging
import utils
import arrow_dec_mpi

def bench_spmm(path: Union[str, None],
               width: int,
               n_features: int,
               iterations: int,
               blocked: bool,
               device: str,
               p_per_side=3,
               ba_neighbors:int=5,
               wandb_api_key: str=None,
               datatype=np.float32):
    assert width > 0

    comm = MPI.COMM_WORLD

    if path is None:
        path = 'test_ba' + "_" + str(p_per_side) + "_" + str(ba_neighbors)
        if comm.Get_rank() == 0:
            # If no path given, generate random graph
            g = igraph.Graph.Barabasi(p_per_side * width, ba_neighbors, 503, directed=False)
            decomposition = arrowdecomposition.arrow_decomposition(g, width, 3, block_diagonal=blocked)
            graphio.save_decomposition(g, decomposition, path, block_diagonal=blocked)
            print("DATASET GENERATED -- ", g.vcount(), " vertices")

        comm.Barrier()

    name = "Arrow_v0.2"
    if blocked:
        name += "_BlockDiagonal"

    wb_logging.wandb_init(comm, path, n_features, iterations, device, name, width, wandb_api_key)

    assert len(path)

    blocks, n_blocks, to_prev, to_next = arrow_dec_mpi.ArrowDecompositionMPI.load_decomposition(comm, path, width, blocked, datatype)

    print("RANK ", comm.Get_rank(), "BLOCKS LOADED", n_blocks, flush=True)

    comm.Barrier()

    if np.sum(n_blocks) == 0:
        # Abort early if no matrix found.
        print("ERROR: Empty Matrix. Check that the file exists and all parameters match (width, block diagonal).", file=sys.stderr)
        return

    if np.sum(2*n_blocks-1) > comm.Get_size():
        print("ERROR: Not enough ranks available. Minimum number of ranks to process the matrix:", np.sum(2*n_blocks-1), file=sys.stderr)
        return

    # Create comms & Allocate processors to block matrices
    arrow = arrow_dec_mpi.ArrowDecompositionMPI.initialize(comm, n_blocks, to_prev, to_next, width, n_features, device, blocked)

    print("RANK ", comm.Get_rank(), "ARROW initialized")

    rank = comm.Get_rank()
    rng = np.random.default_rng(42 + rank)

    comm.Barrier()
    if arrow is not None:

        wb_logging.log({"actual_ranks": arrow.comm.Get_size()})
        if arrow.comm.Get_size() == 0:
            print("Actual size", arrow.comm.Get_size(), flush=True)

        tic = time.perf_counter()

        # Set the A matrix
        print("RANK", comm.Get_rank(), "load from blocks...", flush=True)
        arrow.B.load_sparse_matrix_from_blocks(blocks)

        # Initialize C and X to 0
        arrow.B.zero_rhs(width, n_features)

        # Initialize X (Features)
        if arrow.matrix_index == 0 and arrow.B.is_column_rank():
            X_p0 = 2*rng.random((width, n_features), dtype=datatype)-1
            arrow.B.set_features(X_p0)

        arrow.comm.Barrier()

        toc = time.perf_counter()
        wb_logging.log({"init_time": toc - tic})

        # SPMM
        for i in range(iterations):
            wb_logging.set_iteration_data({"iteration": i})
            tic = time.perf_counter()
            arrow.step()
            toc = time.perf_counter()
            wb_logging.log({"spmm_time": toc-tic})
            print("RANK", comm.Get_rank(), "Iteration", i, " -- ", toc-tic, "s")

    wb_logging.finish()
    comm.Barrier()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark the SpMM')
    parser.add_argument('-f', '--path', type=str, default=None, help='The filename prefix of the decomposed graph. If none, synthetic data is generated.')
    parser.add_argument('-w', '--width', type=int, default=0, help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-c', '--features', type=int, default=16, help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-b', '--blocked', type=utils.str2bool, nargs="?", default=True, help='If true, the matrix has only one block diagonal,')
    parser.add_argument('-i', '--device', type=str, default='gpu', help='Device to use for the MM. Either cpu or gpu.')
    parser.add_argument('-z', '--iterations', type=int, default=1, help='Number of SpMM iteration to run.')
    parser.add_argument('-r', '--ranksperside', type=int, default=3, help='Number of Ranks per Side (For synthetic data only)')
    parser.add_argument('-m', '--ba_neighbors', type=int, default=3, help='Number of neighbors per bertex (For synthetic data only)')
    parser.add_argument('-k', '--wandb_key', type=str, default=None, help='Wandb API key.')

    args = vars(parser.parse_args())
    print(args)

    bench_spmm(args['path'], args['width'], args['features'], args['iterations'], args['blocked'], args['device'], args['ranksperside'], args['ba_neighbors'], args['wandb_key'])
