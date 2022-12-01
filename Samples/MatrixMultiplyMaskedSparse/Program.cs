// #define MATDEBUG

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
            // SmokeTest(3, 2);
            CudaSmokeTest(2000, 1500);
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
        static void SmokeTest(matrix_index n, matrix_index k)
        {
            matrix_data[,] A = CreateSequentialMatrix(n, k);
            matrix_data[,] B = CreateSequentialMatrix(k, k);
            matrix_data[,] P = CreateIndentityMatrix(n, k);

            matrix_data[,] Bt = MatrixTranspose(B);
            matrix_data[,] ABt = MatrixMultiplyNaive(A, Bt);
            matrix_data[,] P_and_ABt = MatrixMask(ABt, P);

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
        static void CudaSmokeTest(matrix_index n, matrix_index k)
        {
            matrix_index band_width = 100;
            matrix_data[,] A = CreateSequentialMatrix(n, k);
            matrix_data[,] B = CreateBandedSequentialMatrix(k, k, band_width);
            matrix_data[,] P = CreateIndentityMatrix(n, k);

            var sw = new Stopwatch();
            
            sw.Restart();
            SparseMatrix sB = new SparseMatrix(B);
            CondensedProductRows cPA = new CondensedProductRows(P, sB, A);
            sw.Stop();
            Console.WriteLine("Finished constructing matricies.");
            Console.WriteLine($"Matrix parsing took: {sw.ElapsedMilliseconds}ms");  



            #if MATDEBUG
            // Debug check
            sw.Restart();
            matrix_data[,] Bt = MatrixTranspose(B);
            matrix_data[,] ABt = MatrixMultiplyNaive(A, Bt);
            matrix_data[,] P_and_ABt = MatrixMask(ABt, P);
            sw.Stop();
            Console.WriteLine("Finished debug check.");
            Console.WriteLine($"Naive multiplication takes: {sw.ElapsedMilliseconds}ms"); 
            #endif
            

            matrix_data[,] PABt = new matrix_data[n, k];

            // Accelerated implementations
            using var context = Context.CreateDefault();
            foreach (var device in context)
            {
                using var accelerator = device.CreateAccelerator(context);

                // Do initial call to compile kernel and warmup
                CondensedMatrixMultiplication(accelerator, cPA, sB, PABt);

                int repeats = 10;

                sw.Restart();
                for(matrix_index i=0; i<repeats; ++i) {
                    CondensedMatrixMultiplication(accelerator, cPA, sB, PABt);
                }
                sw.Stop();

                #if MATDEBUG
                Debug.Assert(MatrixEqual(PABt, P_and_ABt));
                #endif
                
                //Console.WriteLine("PABt:"); PrintMatrix(PABt);
                //Console.WriteLine("P_and_ABt:"); PrintMatrix(P_and_ABt);
                Console.WriteLine($"- Accelerated implementation on {accelerator}: " +
                    $"{(float)sw.ElapsedMilliseconds/(float)repeats}ms");  
            }         
        }

        #endregion   


        #region Accelerated algorithm

        // 
        /// <summary>
        /// Compute PABt = PA * B'
        /// </summary>
        /// <param name="accelerator">The Accelerator to run the multiplication on</param>
        /// <param name="PA">A condensed version of the matrix (P && A) </param>
        /// <param name="B">A sparse matrix Bx</param>
        /// <param name="PABt">The dense matrix to return values into, PA * B'</param>
        static void CondensedMatrixMultiplication(Accelerator accelerator, CondensedProductRows PA, SparseMatrix B, 
            matrix_data [,] PABt) {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<matrix_index, Stride1D.Dense>,
                ArrayView2D<matrix_data, Stride2D.DenseY>,
                ArrayView2D<matrix_data, Stride2D.DenseY>,
                ArrayView1D<matrix_data, Stride1D.Dense>,
                SpecializedValue<matrix_index>>(
                    AcceleratedDotProductKernel
                );

            matrix_index a_row = (matrix_index) PA.m_data.GetLength(0);
            matrix_index row_len = (matrix_index) PA.m_data.GetLength(1);
            matrix_index max_col = (matrix_index) PA.m_data.GetLength(1);
            matrix_index b_row = (matrix_index) B.m_edge_weights.GetLength(0);
        
            using var col_idx = accelerator.Allocate1D(PA.m_col_idx);

            // LIMITATION. It seems that Index2D can only handle int.
            using var rows = accelerator.Allocate2DDenseY<matrix_data>(new Index2D((int) a_row, (int) max_col));
            using var cols = accelerator.Allocate2DDenseY<matrix_data>(new Index2D((int) b_row, (int) max_col));
            using var dotsum = accelerator.Allocate1D<matrix_data>(a_row);

            col_idx.CopyFromCPU(PA.m_col_idx); // do we need this line?
            rows.CopyFromCPU(PA.m_data);
            cols.CopyFromCPU(B.m_edge_weights);

            kernel(dotsum.Extent.ToIntIndex(), col_idx.View, rows.View, cols.View, dotsum.View, SpecializedValue.New(row_len));

            // Copy result to empty dense output matrix
            matrix_data[] h_dotsum = dotsum.GetAsArray1D();
            for(matrix_index i=0; i < a_row; ++i) 
            {
                matrix_index r = PA.m_row_idx[i];
                matrix_index c = PA.m_col_idx[i];
                PABt[r, c] = h_dotsum[i];
            }
        }



        // Form dot product between rows and cols
        // Matching rows to columns via col_idx
        // Return result in dotsum.
        //   where dotsum[x] = rows[x, ] * cols[col_idx[x], ]
        static void AcceleratedDotProductKernel(
            Index1D index,
            ArrayView1D<matrix_index, Stride1D.Dense> col_idx,
            ArrayView2D<matrix_data, Stride2D.DenseY> rows,
            ArrayView2D<matrix_data, Stride2D.DenseY> cols,
            ArrayView1D<matrix_data, Stride1D.Dense> dotsum,
            SpecializedValue<matrix_index> row_len   // This SpecializedValue helps the compiler optimize the loop
        )
        {
            matrix_index row = (matrix_index) index.X;
            matrix_index col = col_idx[row];
            matrix_data sum = 0.0f;

            for (var i = 0; i < row_len; i++)
                sum += rows[row, i] * cols[col, i];

            dotsum[index] = sum;
        }


        #endregion

    } // end class Program
} // end namespace MatrixMultiply
