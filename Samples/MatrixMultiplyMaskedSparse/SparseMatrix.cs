using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using CommunityToolkit.HighPerformance;

namespace MatrixMultiply
{
    using static Utils;
    
    public class SparseMatrix
    {

        // the number of rows, columns.
        int m_nrow, m_ncol;

        // the max # of non-zero entries in m_num_neighbors f
        public int m_f {get;}

        // nrow x f matrix containing column indexes where non-zero values in matrix are for each row in [0:m_nrow]
        public int[,] m_neighbors {get;}

        // vector with number of non-zero entries on each row of m_neighbors
        // for all x, m_neighbors[x, m_num_neighbors[x]:m_f] may contain junk
        public int[] m_num_neighbors {get;}

        // Weights for each entry in m_neighbors
        public float[,] m_edge_weights {get;}


        // Flag to logically transpose the matrix when accessing
        bool m_transposed;


        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <returns>A sparse representation of A</returns>
        public SparseMatrix(float[,] a)
        {
            m_transposed = false;
            m_nrow = a.GetLength(0);
            m_ncol = a.GetLength(1);
            m_num_neighbors = new int[m_nrow];

            // Get counts of the number of columns we need to represent
            m_f = 0;
            for(int i=0; i<m_nrow; ++i) {
                int row_entries = 0;
                for(int j=0; j<m_ncol; ++j) {
                    if(Math.Abs(a[i, j]) > 0.0f) {
                        ++row_entries;
                    }
                }
                m_num_neighbors[i] = row_entries;
                if(row_entries > m_f) {
                    m_f = row_entries;
                }
            }
            Debug.Assert(m_f > 0);

            // Copy non-zero data to sparse storage
            m_neighbors = new int[m_nrow, m_f];
            m_edge_weights = new float[m_nrow, m_f];
            for(int i=0; i<m_nrow; ++i) {
                int idx = 0;
                for(int j=0; j<m_ncol; ++j) {
                    if(Math.Abs(a[i, j]) > 0.0f) {
                        m_neighbors[i, idx] = j;
                        m_edge_weights[i, idx] = a[i, j];
                        ++idx;
                    }
                }
            }
        }

        /// <summary>
        /// Emulate GetLength method from array
        /// </summary>
        /// <param name="dim">Dimension to retrieve inforamation about</param>
        /// <returns>Length of that dimension</returns>
        public int GetLength(int dim) 
        {
            Debug.Assert(dim < 2);
            if(dim == 0) {
                return m_nrow;
            }
            if(dim == 1) {
                return m_ncol;
            }
            return 0;
        }

        /// Logically transpose the matrix
        public void Transpose() {
            m_transposed = !m_transposed;
        }


        // Find the requested column from the original dense matrix 
        // in m_neighbors.
        // Return -1 if that column could not be found.
        // TODO: This is terrible, ask Marcel how to do without copy
        private int FindColumn(int row, int col)
        {
            int nonzero = m_num_neighbors[row];
            Span2D<int> span = m_neighbors;
            // Span2D<int> row_neighbors = span[row, ..nonzero]; // The nonzero column indexes for that row
            Span2D<int> row_neighbors = span.Slice(row, 0, 1, nonzero); // It seems this .NET version does not support range ops as above??
            // Marcel, what I want to do use use BinarySearch in the System assembly.
            // But is this only .NET 7??
            // https://learn.microsoft.com/en-us/dotnet/api/system.memoryextensions.binarysearch?view=net-7.0#system-memoryextensions-binarysearch-2(system-span((-0))-1)
            //
            // int idx = BinarySearch(row_neighbors, col);
            
            // Bad implementation for now
            var span_arr = row_neighbors.ToArray();
            int[] arr = To1DArray(span_arr);
            int idx = Array.BinarySearch(arr, col);
            return idx;
        }


        public float this[int row, int col]
        {
            get
            {
                if(m_transposed) {
                    (row, col) = (col, row);
                }
                int idx = FindColumn(row, col);
                if(idx < 0) {
                    return 0.0f;
                }

                return m_edge_weights[row, idx];
            }
            set
            {
                if(m_transposed) {
                    (row, col) = (col, row);
                }
                int idx = FindColumn(row, col);
                if(idx < 0) {
                    throw new ArgumentException("Index out of bounds when attempting to update sparse matrix");
                }
                m_edge_weights[row, idx] = value;
            }
        }


    } // end SparseMatrix


} // end namespace MatrixMultiply