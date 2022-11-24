using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using CommunityToolkit.HighPerformance;

namespace MatrixMultiply
{
    using static Utils;

    // Store condensed rows from A needed to form C where:
    // C = P && A*B'
    // C = Mask && A*Sparse'
    public class CondensedProductRows
    {
        // condensed rows from A
        public float[,] m_data {get;}

        // Target row in C
        public int[] m_row_idx {get;}

        // Target column in C
        public int[] m_col_idx {get;}

        // Extract condensed rows from A needed to form C where:
        // C = P && A*B'
        // C = Mask && A*Sparse'
        public CondensedProductRows(float[,] Mask, SparseMatrix Sparse, float[,] A)
        {
            int max_col = Sparse.m_f;
            int sparse_rows = Sparse.GetLength(0);

            // TODO with Marcel
            // The logic is probably better done in a single pass via lists
            // But, I'm not sure if ILGPU handles this

            // Count number of non-empty product rows based on Mask && Sparse
            int nrow = 0;
            for(int col=0; col<sparse_rows; ++col) {
                // We are working with Sparse'
                // So, each entry in Sparse.m_num_neighbors is a column for A*[column1, column2, ...]
                int idx_len = Sparse.m_num_neighbors[col];
                for (int row = 0; row < A.GetLength(0); ++row) {
                    if(Mask[row, col] != 0.0f && idx_len > 0) {
                        ++nrow;
                    }
                }
            }

            // Allocate storage and copy relevant data
            m_data = new float[nrow, max_col];
            m_row_idx = new int[nrow];
            m_col_idx = new int[nrow];
            int data_row = 0;
            for(int col=0; col<sparse_rows; ++col) {
                int idx_len = Sparse.m_num_neighbors[col];
                for (int row = 0; row < A.GetLength(0); ++row) {
                    if(Mask[row, col] == 0.0f) {
                        continue;
                    }
                    for (int j = 0; j < idx_len; ++j) {
                        m_data[data_row, j] = A[row, Sparse.m_neighbors[col, j]];
                    }
                    if (idx_len > 0) {
                        m_row_idx[data_row] = row;
                        m_col_idx[data_row] = col;
                        ++data_row;
                    }
                }
            }
        }


        // Print expected dot products and output locations from P && A * B'
        public void PrintABtDots(SparseMatrix B)
        {
            int nrow = m_row_idx.GetLength(0);
            for(int i=0; i<nrow; ++i) {
                int r = m_row_idx[i];
                int c = m_col_idx[i];
                Console.WriteLine($"[{r}, {c}]:");
                int idx_len = B.m_f;  // Debug, check the 0 padding on rows

                // print CondensedProductRows data
                Console.Write("[");
                for(int j=0; j<idx_len; ++j) {
                    Console.Write($"{m_data[i, j]}, ");
                }
                Console.Write("]");

                Console.Write(" . ");

                // Show row from B
                Console.Write("[");
                for(int j=0; j<idx_len; ++j) {
                    Console.Write($"{B.m_edge_weights[c, j]}, ");
                }
                Console.Write("]");

                // Show dot product
                Console.Write(" = ");
                float sum = 0;
                for(int j=0; j<idx_len; ++j) {
                    sum += m_data[i, j] * B.m_edge_weights[c, j];
                }
                Console.WriteLine(sum);
                Console.WriteLine();
            }
        }

    } // end class CondensedProductRows

} // end namespace MatrixMultiply


