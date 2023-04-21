import argparse

import numpy as np

import graphio
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute the number of blocks for a given decomposition')
    parser.add_argument('-p', '--path', type=str, default=None, help='The filename prefix of the decomposed graph.')
    parser.add_argument('-w', '--width', type=int, default=0, help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-b', '--blocked', type=utils.str2bool, nargs="?", default=True, help='If true, the matrix has only one block diagonal,')

    args = vars(parser.parse_args())

    print(args)

    width = args['width']
    decomposition = graphio.load_decomposition(args['path'], width, args['blocked'])

    block_count : np.ndarray = np.zeros(len(decomposition))
    print(decomposition)
    for i, (A, p) in enumerate(decomposition):
        blocks = graphio.split_matrix_to_blocks(A, width, use_min_shape=False)

        for row in blocks:
            for b in row:
                if b is not None and b.nnz > 0:
                    block_count[i] += 1
                    break


    print("Nonzero block rows", block_count)
    block_count_2 = block_count.copy()
    block_count *= 2
    block_count -= 1
    print("Nodes required", block_count)
    print("IN TOTAL: ", np.sum(block_count))

    block_count_2 = block_count_2 * 3 - 1
    print("NONZERO BLOCKS", block_count_2)
    print("NNZ BLOCKS IN TOTAL", np.sum(block_count_2))