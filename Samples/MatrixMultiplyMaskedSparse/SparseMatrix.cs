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
        matrix_index m_nrow, m_ncol;

        // the max # of non-zero entries in m_num_neighbors f
        public matrix_index m_f {get;}

        // nrow x f matrix containing column indexes where non-zero values in matrix are for each row in [0:m_nrow]
        public matrix_index[,] m_neighbors {get;}

        // vector with number of non-zero entries on each row of m_neighbors
        // for all x, m_neighbors[x, m_num_neighbors[x]:m_f] may contain junk
        public matrix_index[] m_num_neighbors {get;}

        // Weights for each entry in m_neighbors
        public matrix_data[,] m_edge_weights {get;}


        /// <summary>
        /// Construct a sparse matrix of a dense matrix.
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <returns>A sparse representation of A</returns>
        public SparseMatrix(matrix_data[,] a)
        {
            m_nrow = (matrix_index)a.GetLength(0);
            m_ncol = (matrix_index)a.GetLength(1);
            m_num_neighbors = new matrix_index[m_nrow];

            // Get counts of the number of columns we need to represent
            m_f = 0;
            for(matrix_index i=0; i<m_nrow; ++i) {
                matrix_index row_entries = 0;
                for(matrix_index j=0; j<m_ncol; ++j) {
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
            m_neighbors = new matrix_index[m_nrow, m_f];
            m_edge_weights = new matrix_data[m_nrow, m_f];
            for(matrix_index i=0; i<m_nrow; ++i) {
                matrix_index idx = 0;
                for(matrix_index j=0; j<m_ncol; ++j) {
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
        public matrix_index GetLength(matrix_index dim) 
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


        // Find the requested column from the original dense matrix 
        // in m_neighbors.
        // Return -1 if that column could not be found.
        // TODO: This is terrible, ask Marcel how to do without copy
        private matrix_index FindColumn(matrix_index row, matrix_index col)
        {
            matrix_index nonzero = m_num_neighbors[row];
            var row_neighbors = m_neighbors.Slice(row, nonzero);
            matrix_index idx = row_neighbors.BinarySearch(col);
            return idx;
        }


        public matrix_data this[matrix_index row, matrix_index col]
        {
            get
            {
                try 
                {
                    matrix_index idx = FindColumn(row, col);
                    return m_edge_weights[row, idx];
                } 
                catch (ArgumentOutOfRangeException)
                {       
                    return 0.0f;
                }
            }
            set
            {
                matrix_index idx = FindColumn(row, col);
                if(idx < 0) {
                    throw new ArgumentException("Index out of bounds when attempting to update sparse matrix");
                }
                m_edge_weights[row, idx] = value;
            }
        }


    } // end SparseMatrix


} // end namespace MatrixMultiply