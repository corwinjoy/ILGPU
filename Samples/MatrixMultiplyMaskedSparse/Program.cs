/* Problem statement:
 * Create a CUDA kernal for the following sparse matrix problem
 * C = P && A*B'
 * where
 * A = n x k, dense matrix
 * B = k x k, sparse matrix
 * P = n x k, dense matrix of 1s and 0s indicating if we care about an entry in the product A*B'
 *
 * The sparse format for B is:
 * numNeighbors[0:k] = an int valued vector containing the number of non-zero values in each row, where number < f
 * neighbors[0:k, 0:f] = column indexes for the non-zero values in B.
 * for each i=0...k-1, neighbors[i, 0:numNeighbors[i]] contain valid indices and neighbors[i, numNeighbors[i]:_] has noise
 * edgeWeights[0:k, 0:f] contains values for the non-zero entries in B, indexed in the same way as above
 *
 * Reasonable value ranges are n=5-300, k=10 000-50 000, f=5-5000, P typically has 5-10 values in each column.
 *
 *
*/

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace MatrixMultiply
{
    using static Utils;

    class Program
    {
        /// <summary>
        /// Main entry point
        /// </summary>
        static void Main()
        {
            SmokeTest(3, 2);
        }

        #region Helper functions

        /*
        * Compute
        *
        * C = P && A*B' using matrix operations
        *
        * where
        * A = n x k, dense matrix
        * B = k x k, sparse matrix
        * P = n x k, dense matrix of 1s and 0s indicating if we care about an entry in the product A*B'
        */
        static void SmokeTest(int n, int k)
        {
            float[,] A = CreateSequentialMatrix(n, k);
            float[,] B = CreateSequentialMatrix(k, k);
            float[,] P = CreateIndentityMatrix(n, k);

            float[,] Bt = MatrixTranspose(B);
            float[,] ABt = MatrixMultiplyNaive(A, Bt);
            float[,] P_and_ABt = MatrixMask(ABt, P);

            Console.WriteLine("***************************************************************\n");
            Console.WriteLine("A:"); PrintMatrix(A);
            Console.WriteLine("B:"); PrintMatrix(B);
            Console.WriteLine("P:"); PrintMatrix(P);
            Console.WriteLine("Bt:"); PrintMatrix(Bt);
            Console.WriteLine("ABt:"); PrintMatrix(ABt);
            Console.WriteLine("P && ABt:"); PrintMatrix(P_and_ABt);
            Console.WriteLine("Basic Tests Done!");

            Console.WriteLine("***************************************************************\n");
            Console.WriteLine("Sparse matrix tests:");

            SparseMatrix sP = new SparseMatrix(P);
            Console.WriteLine("sP:"); PrintMatrix(sP);

            SparseMatrix sB = new SparseMatrix(B);
            Console.WriteLine("sB:"); PrintMatrix(sB);
            Console.WriteLine("Sparse Tests Done!");

            Console.WriteLine("***************************************************************\n");
            Console.WriteLine("Condensed Row Tests:");

            CondensedProductRows CPR = new CondensedProductRows(P, sB, A);
            Console.WriteLine("CPR row data:"); PrintMatrix(CPR.m_data);
            Console.WriteLine("CPR dot products:"); CPR.PrintABtDots(sB);
            Console.WriteLine("Condensed Row Tests Done!");
            Console.WriteLine("***************************************************************\n");
        }  

        #endregion   

    } // end class Program
} // end namespace MatrixMultiply
