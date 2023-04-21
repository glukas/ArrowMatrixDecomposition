from typing import List

from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

import graphio
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _blocks_to_matrix(blocks):
    structure_matrix = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    max_nnz = 0
    for row in range(len(blocks)):
        for col in range(len(blocks)):
            if blocks[row][col] is not None:
                b: sparse.csr_matrix = blocks[row][col]
                if b.nnz > 0:
                    #print(b.nnz)
                    structure_matrix[row][col] = b.nnz
                    max_nnz = max(b.nnz, max_nnz)

    return structure_matrix, max_nnz


def visualize_structure_matrices(structure_matrices: List[np.ndarray], vmax=None):
    fig, axs = plt.subplots(1, len(structure_matrices))
    #fig.set_figheight(7)
    #fig.set_figwidth
    im = None
    for i, S in enumerate(structure_matrices):
        im = axs[i].imshow(S, vmax=vmax)

    #fig.colorbar(im)
    plt.show()

def display_sns(structure_matrices: List[np.ndarray], vmax, normalize=5000000):

    dfs = [pd.DataFrame(A/normalize) for A in structure_matrices]


    max_value = vmax/normalize
    ratios = [1.0 for i in range(len(structure_matrices))]
    #ratios.append(0.05)

    fig, axs = plt.subplots(ncols=len(structure_matrices),
                            gridspec_kw= dict(width_ratios=ratios))

    fig.set_figheight(5)
    fig.set_figwidth(5*len(structure_matrices))

    # Define the levels and colors for the discrete steps
    levels = np.arange(0.0, max_value, 0.1)
    colors = plt.cm.get_cmap('Blues', len(levels))(np.arange(len(levels)))

    # Create the discrete colormap
    cmap = ListedColormap(colors)
    # Create the heatmap
   # heatmap = plt.imshow(data, cmap=cmap, norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True))

    # Add colorbar
    #cbar = plt.colorbar(heatmap, ticks=levels)
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    #palette = sns.cubehelix_palette(as_cmap=True)
    palette = sns.color_palette("Blues", as_cmap=True)
    for i in range(len(dfs)):
        ax = axs
        #sns.heatmap(dfs[i], cmap=palette, square=True, annot=False, ax=ax, vmin=0, vmax=1,
        #            cbar_kws={'location': 'right', 'shrink': 0.8, 'format': "{x:.2f}"},
        #            yticklabels=[], xticklabels=[], norm=LogNorm(),
        #            linewidths=0.1, linecolor='grey')

        sns.heatmap(dfs[i], cmap=cmap, square=True, annot=False, ax=ax,
                    cbar_kws={'location': 'right', 'shrink': 0.8, 'format': "{x:.2f}"},
                    yticklabels=[], xticklabels=[], norm=LogNorm(),
                    linewidths=0.1, linecolor='grey')


    #sns.heatmap(dfs[len(dfs)-1], annot=True, yticklabels=False, cbar=False, ax=axs[len(dfs)-1], vmax=vmax)

    #cbar = fig.colorbar(axs[0].collections[0], cax=axs[1], ticks=[])

    #plt.show()
    return fig, axs

def display_block_structure_of_matrices(decomposition: List[sparse.csr_matrix], width:int, savefile=None):
    structure = []
    total_nnz = 0
    max_nnz = 0
    for A in decomposition:
        blocks = graphio.split_matrix_to_blocks(A, width, use_min_shape=False)
        B, max_nnz_a = _blocks_to_matrix(blocks)
        structure.append(B)
        total_nnz += A.getnnz()
        max_nnz = max(max_nnz, max_nnz_a)

    print(decomposition)
    assert len(decomposition) > 0
    fig, axs = display_sns(structure, max_nnz)

    if savefile is not None:
        print("SAVING", savefile + '.pdf')
        plt.savefig(savefile + '.pdf', format='pdf')
        plt.savefig(savefile + '.svg', format='svg')
        plt.savefig(savefile + '.png', format='png')
    else:
        plt.plot()

    #visualize_structure_matrices(structure, max_nnz)

def display_block_structure(base_path: str, width: int, is_block_diagonal: bool, save_file: str = None):
    decomposition = graphio.load_decomposition(base_path, width, block_diagonal=is_block_diagonal)
    display_block_structure_of_matrices([a for a, p in decomposition], width, save_file)
