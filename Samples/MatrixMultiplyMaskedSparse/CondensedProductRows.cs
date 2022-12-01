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
        public matrix_data[,] m_data {get;}

        // Target row in C
        public matrix_index[] m_row_idx {get;}

        // Target column in C
        public matrix_index[] m_col_idx {get;}

        // Extract condensed rows from A needed to form C where:
        // C = P && A*B'
        // C = Mask && A*Sparse'
        public CondensedProductRows(matrix_data[,] Mask, SparseMatrix Sparse, matrix_data[,] A)
        {
            matrix_index max_col = Sparse.m_f;
            matrix_index sparse_rows = Sparse.GetLength(0);

            // TODO with Marcel
            // The logic is probably better done in a single pass via lists
            // But, I'm not sure if ILGPU handles this

            // Count number of non-empty product rows based on Mask && Sparse
            matrix_index nrow = 0;
            for(matrix_index col=0; col<sparse_rows; ++col) {
                // We are working with Sparse'
                // So, each entry in Sparse.m_num_neighbors is a column for A*[column1, column2, ...]
                matrix_index idx_len = Sparse.m_num_neighbors[col];
                for (matrix_index row = 0; row < A.GetLength(0); ++row) {
                    if(Mask[row, col] != 0.0f && idx_len > 0) {
                        ++nrow;
                    }
                }
            }

            // Allocate storage and copy relevant data
            m_data = new matrix_data[nrow, max_col];
            m_row_idx = new matrix_index[nrow];
            m_col_idx = new matrix_index[nrow];
            matrix_index data_row = 0;
            for(matrix_index col=0; col<sparse_rows; ++col) {
                matrix_index idx_len = Sparse.m_num_neighbors[col];
                for (matrix_index row = 0; row < A.GetLength(0); ++row) {
                    if(Mask[row, col] == 0.0f) {
                        continue;
                    }
                    for (matrix_index j = 0; j < idx_len; ++j) {
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
            matrix_index nrow = (matrix_index) m_row_idx.GetLength(0);
            for(matrix_index i=0; i<nrow; ++i) {
                matrix_index r = m_row_idx[i];
                matrix_index c = m_col_idx[i];
                Console.WriteLine($"[{r}, {c}]:");
                matrix_index idx_len = B.m_f;  // Debug, check the 0 padding on rows

                // print CondensedProductRows data
                Console.Write("[");
                for(matrix_index j=0; j<idx_len; ++j) {
                    Console.Write($"{m_data[i, j]}, ");
                }
                Console.Write("]");

                Console.Write(" . ");

                // Show row from B
                Console.Write("[");
                for(matrix_index j=0; j<idx_len; ++j) {
                    Console.Write($"{B.m_edge_weights[c, j]}, ");
                }
                Console.Write("]");

                // Show dot product
                Console.Write(" = ");
                matrix_data sum = 0;
                for(matrix_index j=0; j<idx_len; ++j) {
                    sum += m_data[i, j] * B.m_edge_weights[c, j];
                }
                Console.WriteLine(sum);
                Console.WriteLine();
            }
        }

    } // end class CondensedProductRows

} // end namespace MatrixMultiply


