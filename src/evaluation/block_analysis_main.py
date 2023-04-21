import argparse
import graphio
import block_analysis
import utils
from scipy import sparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the nonzero block structure')
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='The filename prefix of the decomposed graph. If none, synthetic data is generated.')
    parser.add_argument('-w', '--width', type=int, default=5000000, help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-b', '--blocked', type=utils.str2bool, nargs="?", default=True,
                        help='If true, the matrix has only one block diagonal,')

    parser.add_argument('-o', '--original', type=bool, nargs="?", default=False,
                        help='If true, the original matrix is shown,')

    args = vars(parser.parse_args())
    print(args)

    decomposition = graphio.load_decomposition(args['path'], args['width'], args['blocked'], no_permutation=True)

    filename = args['path'].split('/')[-1]

    for i in range(len(decomposition)):
        block_analysis.display_block_structure_of_matrices([decomposition[i][0]], args['width'],
                                                           './figures_first/' + filename + "_" + str(i))
