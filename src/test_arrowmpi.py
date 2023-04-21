import sys
import unittest
import igraph
import numpy as np
import arrow_dec_mpi
from preprocessing import arrowdecomposition
import arrow_mpi
from mpi4py import MPI
import arrow_main_mpi
import graphio
import test_arrowdecomposition

"""
The following unit tests require MPI to work correctly.
To be able to run all tests, a number of ranks 2x for an odd number x is required.
To be able to run the largest tests, at least 28 ranks are required.
In conclusion, to run all tests, type:
mpiexec -n 30 --oversubscribe python test_arrowmpi.py
"""


class TestArrowMPI(unittest.TestCase):

    def test_all_to_all(self):

        # Test inverse permutation (easy)
        ranks = 2
        prev_ranks = 6
        rows_per_rank = 4
        cols = 6
        permutation = np.asarray(list(reversed(range(ranks*rows_per_rank))))
        print(permutation)
        for i in range(ranks):
            slice = permutation[i*rows_per_rank : (i+1)*rows_per_rank]
            counts, displs, p, out_p = arrow_dec_mpi.ArrowDecompositionMPI._all_to_all_tables(slice, rows_per_rank, cols, prev_ranks + ranks, prev_ranks)
            print(counts, displs, p, out_p)

            self.assertEqual(rows_per_rank*cols, counts[ranks+prev_ranks-i-1])
            self.assertEqual(rows_per_rank*cols, np.sum(counts))
            self.assertEqual(0, displs[ranks+prev_ranks-i-1])

            counts, displs, p, out_p = arrow_dec_mpi.ArrowDecompositionMPI._all_to_all_tables(slice, rows_per_rank, cols, ranks + prev_ranks, 0)
            print(counts, displs, p, out_p)
            self.assertEqual(rows_per_rank*cols, counts[ranks-i-1])
            self.assertEqual(rows_per_rank*cols, np.sum(counts))
            self.assertEqual(0, displs[ranks-i-1])


        # Test random permutation
        permutation = 2*np.asarray(list(range(ranks * rows_per_rank)))
        np.random.shuffle(permutation)
        print(permutation)
        full_permutation = np.concatenate([permutation, np.arange(ranks*rows_per_rank,  ranks*rows_per_rank+prev_ranks*rows_per_rank-permutation.size)])
        print(full_permutation)
        inverse_permutation = np.argsort(full_permutation)
        inverse_permutation = np.minimum(inverse_permutation, ranks*rows_per_rank)
        print(inverse_permutation)
        assert(inverse_permutation.size == prev_ranks*rows_per_rank)

        for i in range(ranks):
            slice = permutation[i * rows_per_rank: (i + 1) * rows_per_rank]
            print("SLICE", slice)
            counts, displs, p, out_p = arrow_dec_mpi.ArrowDecompositionMPI._all_to_all_tables(slice, rows_per_rank, cols, ranks + prev_ranks, 0)

            ranks_after_permuting_np = np.floor_divide(slice[p], rows_per_rank)
            ranks_after_permuting = list(ranks_after_permuting_np)
            golden_sort = sorted(np.floor_divide(slice, rows_per_rank))
            self.assertSequenceEqual(golden_sort, ranks_after_permuting)

            print(counts, displs, p)
            for j in range(ranks):
                c = np.count_nonzero(ranks_after_permuting_np == j)
                self.assertEqual(c*cols, counts[j])

        for i in range(prev_ranks):
            inverse_slice = inverse_permutation[i * rows_per_rank: (i + 1) * rows_per_rank]
            print("SLICE", inverse_slice)

            f_counts, f_displs, f_p, f_out_p = arrow_dec_mpi.ArrowDecompositionMPI._all_to_all_tables(inverse_slice, rows_per_rank, cols, prev_ranks + ranks, prev_ranks)

            self.assertEqual(0, f_displs[prev_ranks])
            print(f_counts, f_displs, f_p, f_out_p)

            ranks_after_permuting_np = np.floor_divide(inverse_slice[f_p], rows_per_rank)
            ranks_after_permuting = list(ranks_after_permuting_np)
            golden_sort = sorted(np.floor_divide(inverse_slice, rows_per_rank))
            self.assertSequenceEqual(golden_sort, ranks_after_permuting)

            print("RANKS after Permuting", ranks_after_permuting)

            for j in range(ranks):
                c = np.count_nonzero(ranks_after_permuting_np == j)
                self.assertEqual(c*cols, f_counts[prev_ranks+j])

    def test_decomposition(self):
        comm = MPI.COMM_WORLD

        if comm.Get_size() < 6:
            return

        assert comm.Get_size() % 2 == 0

        np.random.seed(955)

        half = comm.Get_size() // 2

        if half % 2 == 0:
            print("WARNING: To execute test_decomposition, a number of ranks 2x for odd x is required. Skipping test.", file=sys.stderr)
            return

        assert half % 2 == 1

        group1 = MPI.Group.Range_excl(comm.Get_group(), [(half, comm.Get_size()-1, 1)])
        group2 = MPI.Group.Range_incl(comm.Get_group(), [(half, comm.Get_size()-1, 1)])

        assert group1.Get_size()+group2.Get_size() == comm.Get_size()

        b = 100
        k = 10
        n = b*((group1.Get_size()+1)//2)

        #X = np.zeros((n, k), dtype=np.float32)
        rng = np.random.default_rng(875)

        X = rng.random((n, k), dtype=np.float32)
        # Test result
        B1 = self.get_arrow_matrix(comm, n, 15)
        #B2 = self.get_arrow_matrix(comm, n, b)
        #B1 = np.eye(n)
        B2 = np.eye(n, dtype=np.float32)

        # Reverse Permutation
        permutation = np.asarray(list(reversed(range(n))), dtype=np.int64)
        #permutation = np.asarray([2, 3, 5, 6, 0, 1, 4, 5])
        np.random.shuffle(permutation)
        #print("PERMUTATION", permutation)
        inverse_permutation = np.argsort(permutation)

        #print("INV PERMUTATION", inverse_permutation)
        matrix1 = None
        matrix2 = None

        #blocks = [group1.Get_size(), group2.Get_size()]
        groups = [group1, group2]
        if comm.rank < half:
            matrix1 = arrow_mpi.ArrowMPI(MPI.Comm.Create(comm, group1))
            matrix1.zero_rhs(b, k)

            matrix1._distribute_arrays(B1, X, b, 0)

            sliced_p = None
            if matrix1.is_column_rank():
                sliced_p = inverse_permutation[matrix1.column_comm.rank*b: (matrix1.column_comm.rank+1)*b]

            runner = arrow_dec_mpi.ArrowDecompositionMPI(comm, matrix1, 0, b, k, groups, None, sliced_p)

        else:
            matrix2 = arrow_dec_mpi.ArrowMPI(MPI.Comm.Create(comm, group2))
            matrix2.zero_rhs(b, k)

            X2 = X[permutation]
            matrix2._distribute_arrays(B2, X2, b, 0)

            sliced_p = None
            if matrix2.is_column_rank():
                sliced_p = permutation[matrix2.column_comm.rank*b: (matrix2.column_comm.rank+1)*b]

            runner = arrow_dec_mpi.ArrowDecompositionMPI(comm, matrix2, 1, b, k, groups, sliced_p, None)

        X2 = self._iterate_and_test(comm, B1, B2, X, matrix1, matrix2, runner, permutation, inverse_permutation)
        X3 = self._iterate_and_test(comm, B1, B2, X2, matrix1, matrix2, runner, permutation, inverse_permutation)
        X4 = self._iterate_and_test(comm, B1, B2, X3, matrix1, matrix2, runner, permutation, inverse_permutation)

        comm.Barrier()

    def test_load_graph_distributed(self):

        comm = MPI.COMM_WORLD
        rank = comm.rank
        p = comm.Get_size()

        np.random.seed(459)

        b_s = [b for b in range(5, 20)]
        block_diagonal = False
        if rank == 0:
            dataset = [(igraph.Graph.Barabasi(3 * b, 3, 503), b) for b in b_s]

            for g, b in dataset:
                decomposition = arrowdecomposition.arrow_decomposition(g, b, 100, block_diagonal=block_diagonal)
                graphio.save_decomposition(g, decomposition, 'test_ba', block_diagonal=block_diagonal)

        comm.Barrier()

        for b in b_s:

            blocks = arrow_dec_mpi.ArrowDecompositionMPI.load_decomposition(comm, 'test_ba', b, is_block_diagonal=block_diagonal)
            comm.Barrier()

            print("RANK", comm.rank, " -- ", blocks, flush=True)

    def test_decomposition_on_graph(self):

        comm = MPI.COMM_WORLD
        rank = comm.rank
        p = comm.Get_size()
        block_diagonal = True
        np.random.seed(459)

        b_s = [b for b in range(2, 15)]

        factor = 3

        if rank == 0:
            dataset = [(igraph.Graph.Barabasi(factor * b, 3, 503, directed=False), b) for b in b_s]

            for g, b in dataset:
                decomposition = arrowdecomposition.arrow_decomposition(g, b, 100, block_diagonal=block_diagonal)

                for a in decomposition:
                    a.permutation = list(range(len(a.permutation)))

                graphio.save_decomposition(g, decomposition, 'test_ba', use_width=True, block_diagonal=block_diagonal)

        comm.Barrier()

        for b in b_s:

            k = 4

            blocks, n_blocks, to_prev, to_next = arrow_dec_mpi.ArrowDecompositionMPI.load_decomposition(comm, 'test_ba', b, block_diagonal)

            comm.Barrier()

            n = n_blocks[0] * b

            # Create comms & Allocate processors to block matrices
            arrow: arrow_dec_mpi.ArrowDecompositionMPI = arrow_dec_mpi.ArrowDecompositionMPI.initialize(comm, n_blocks, to_prev, to_next, b, k)

            decomposition = graphio.load_decomposition('test_ba', b, block_diagonal=block_diagonal)
            permutations = [p for _, p in decomposition]

            comm.Barrier()

            rng = np.random.default_rng(42)
            #X = np.arange(n * k).reshape((n, k)).astype(np.float32)
            X = rng.random((n, k), dtype=np.float32)

            if rank == 0:
                print("X", X)

            if arrow is not None:

                # Set the A matrix
                arrow.B.load_sparse_matrix_from_blocks(blocks)

                # Generate and set X
                # Create X
                arrow.B.zero_rhs(b, k)

                if arrow.matrix_index == 0 and arrow.B.is_column_rank():
                    print(permutations[0])
                    X_p0 = X[permutations[0]]
                    arrow.B.set_features_slice_from_features(X_p0)

            comm.Barrier()

            if arrow is not None:

                arrow.step()
                arrow._propagate_features()

                if arrow.matrix_index == 0:
                    if rank == 0:
                        print("RANK", comm.rank, "X", X)

                    # C is in the permuted order, so we permute golden_C
                    golden_C = test_arrowdecomposition.TestArrowDecomposition.compute_spmm(decomposition, X)[permutations[0]]

                    C = np.zeros_like(golden_C, dtype=np.float32)

                    arrow.B.allgather_result(C)

                    if rank == 0:
                        error = C-golden_C
                        max_error = np.max(error)
                        print("MAX DIFF", max_error)
                        #if max_error > 0.1:
                        print("DIFF:", error)
                        print("RANK", comm.rank, "GOLDEN C", golden_C)
                        print("RANK", comm.rank, "ACTUAL C", C)

                    self.assertTrue(np.allclose(C, golden_C))
                else:
                    # C is in the permuted order, so we permute golden_C to the order of matrix index 1
                    golden_C = test_arrowdecomposition.TestArrowDecomposition.compute_spmm(decomposition, X)[permutations[1]]
                    if arrow.B.is_column_rank():
                        self.assertTrue(np.allclose(arrow.B.C_i, golden_C[b*arrow.B.column_comm.rank:b*(arrow.B.column_comm.rank+1)]))

    def _iterate_and_test(self, comm, B1, B2, X, matrix1, matrix2, runner, permutation, inverse_permutation):

        runner.step()
        runner._propagate_features()

        golden_C = B1 @ X + (B2 @ X[permutation])[inverse_permutation]
        comm.Bcast(golden_C, 0)
        C = np.zeros_like(golden_C, dtype=np.float32)

        if comm.rank < comm.Get_size() // 2:
            assert matrix1 is not None

            matrix1.allgather_result(C)

            if comm.rank == 0:
                print("M1 GOLDEN C:", golden_C)
                print("M1 ACTUAL C:", C)
                print("M1 DIFF:", C - golden_C)
                self.assertTrue(np.allclose(C, golden_C))
        else:
            matrix2.allgather_result(C)

            if comm.rank == comm.Get_size() // 2:
                golden_C = golden_C[permutation]
                print("M2 GOLDEN C:", golden_C)
                print("M2 ACTUAL C:", C)
                print("M2 DIFF:", C - golden_C)
                self.assertTrue(np.allclose(C, golden_C))

        return golden_C

    def test_spmm(self):

        world_comm = MPI.COMM_WORLD

        if world_comm.Get_size() < 3:
            return

        if world_comm.Get_size() % 2 == 0:
            # Exclude last rank so it's odd

            is_last = world_comm.rank == world_comm.Get_size()-1
            comm = world_comm.Create(world_comm.Get_group().Range_excl([(world_comm.Get_size()-1, world_comm.Get_size()-1, 1)]))

            if is_last:
                world_comm.Barrier()
                return
        else:
            comm = world_comm

        runner = arrow_mpi.ArrowMPI(comm)

        k = 3
        b = 6
        n = b*((comm.Get_size()+1)//2)
        X = np.zeros((n, k), dtype=np.float32)
        rng = np.random.default_rng(42)

        A = self.get_arrow_matrix(comm, n, b)

        if comm.rank == 0:
            # Random X
            X = rng.random((n, k), dtype=np.float32)

            print("A: ", A)
            print("X: ", X)

        runner._distribute_arrays(A, X, b, 0)
        runner.spmm()

        golden_C = A @ X
        C = np.zeros_like(golden_C)
        if comm.rank == 0:
            print("GOLDEN C:", golden_C)
        assert(golden_C.shape == C.shape)

        runner.allgather_result(C)

        if comm.rank == 0:
            print("ACTUAL C:", C)
            print("DIFF:", C-golden_C)

        self.assertTrue(np.allclose(C, golden_C))

        world_comm.Barrier()

    def get_arrow_matrix(self, comm, n, b):
        A = np.zeros((n, n), dtype=np.float32)

        if comm.rank == 0:
            # Structured A
            A[0:b, :] = 1
            A[:, 0:b] = 1
            A[1, 0:b] = 0
            A[2, b:] = 0
            A[:, 3] = 0
            for i in range(b, n - 1):
                A[i, i - 1] = 1
                A[i, i - 3] = 1
                A[i, i - b] = 1
                A[i, i - b + 1] = 1
                A[i, i + 1] = 1
                if i + 3 < n:
                    A[i, i + 3] = 1
                if i + b < n:
                    A[i, i + b - 1] = 1
                    A[i, i + b] = 1

        return A

    def test_larger_ranks(self):

        if MPI.COMM_WORLD.Get_size() < 24:
            print("WARNING: the number of ranks is not big enough to run test_larger_ranks. Skipping Test.")
            return

        arrow_main_mpi.bench_spmm(None, 100, 2, 2, True, 'cpu', p_per_side=4, ba_neighbors=8)
        arrow_main_mpi.bench_spmm(None, 1000, 5, 2, True, 'cpu', p_per_side=4, ba_neighbors=7)
        arrow_main_mpi.bench_spmm(None, 2000, 32, 2, True, 'cpu', p_per_side=4, ba_neighbors=7)
        arrow_main_mpi.bench_spmm(None, 1000, 64, 3, True, 'cpu', p_per_side=4, ba_neighbors=9)

    def test_kmer(self):

        if MPI.COMM_WORLD.Get_size() < 28:
            print("WARNING: the number of ranks is not big enough to run test_kmer. Skipping Test.")
            return

        arrow_main_mpi.bench_spmm('../GraphDatasets/processed/kmer_U1a', 5000000, 2, 2, True, 'cpu')


if __name__ == '__main__':
    tester = TestArrowMPI()

    if MPI.COMM_WORLD.Get_rank() == 0:
        tester.test_all_to_all()

    MPI.COMM_WORLD.Barrier()

    tester.test_spmm()
    tester.test_decomposition()
    tester.test_decomposition_on_graph()
    tester.test_larger_ranks()
    tester.test_kmer()