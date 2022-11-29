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
        static void CudaSmokeTest(int n, int k)
        {
            int band_width = 100;
            float[,] A = CreateSequentialMatrix(n, k);
            float[,] B = CreateBandedSequentialMatrix(k, k, band_width);
            float[,] P = CreateIndentityMatrix(n, k);

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
            float[,] Bt = MatrixTranspose(B);
            float[,] ABt = MatrixMultiplyNaive(A, Bt);
            float[,] P_and_ABt = MatrixMask(ABt, P);
            sw.Stop();
            Console.WriteLine("Finished debug check.");
            Console.WriteLine($"Naive multiplication takes: {sw.ElapsedMilliseconds}ms"); 
            #endif
            

            float[,] PABt = new float[n, k];

            // Accelerated implementations
            using var context = Context.CreateDefault();
            foreach (var device in context)
            {
                using var accelerator = device.CreateAccelerator(context);

                // Do initial call to compile kernel and warmup
                CondensedMatrixMultiplication(accelerator, cPA, sB, PABt);

                int repeats = 10;

                sw.Restart();
                for(int i=0; i<repeats; ++i) {
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
            float [,] PABt) {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseY>,
                ArrayView2D<float, Stride2D.DenseY>,
                ArrayView1D<float, Stride1D.Dense>,
                SpecializedValue<int>>(
                    AcceleratedDotProductKernel
                );

            int a_row = PA.m_data.GetLength(0);
            int row_len = PA.m_data.GetLength(1);
            int max_col = PA.m_data.GetLength(1);
            int b_row = B.m_edge_weights.GetLength(0);
        
            using var col_idx = accelerator.Allocate1D(PA.m_col_idx);
            using var rows = accelerator.Allocate2DDenseY<float>(new Index2D(a_row, max_col));
            using var cols = accelerator.Allocate2DDenseY<float>(new Index2D(b_row, max_col));
            using var dotsum = accelerator.Allocate1D<float>(a_row);

            col_idx.CopyFromCPU(PA.m_col_idx); // do we need this line?
            rows.CopyFromCPU(PA.m_data);
            cols.CopyFromCPU(B.m_edge_weights);

            kernel(dotsum.Extent.ToIntIndex(), col_idx.View, rows.View, cols.View, dotsum.View, SpecializedValue.New(row_len));

            // Copy result to empty dense output matrix
            float[] h_dotsum = dotsum.GetAsArray1D();
            for(int i=0; i < a_row; ++i) 
            {
                int r = PA.m_row_idx[i];
                int c = PA.m_col_idx[i];
                PABt[r, c] = h_dotsum[i];
            }
        }



        // Form dot product between rows and cols
        // Matching rows to columns via col_idx
        // Return result in dotsum.
        //   where dotsum[x] = rows[x, ] * cols[col_idx[x], ]
        static void AcceleratedDotProductKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> col_idx,
            ArrayView2D<float, Stride2D.DenseY> rows,
            ArrayView2D<float, Stride2D.DenseY> cols,
            ArrayView1D<float, Stride1D.Dense> dotsum,
            SpecializedValue<int> row_len   // This SpecializedValue helps the compiler optimize the loop
        )
        {
            int row = index.X;
            int col = col_idx[row];
            float sum = 0.0f;

            for (var i = 0; i < row_len; i++)
                sum += rows[row, i] * cols[col, i];

            dotsum[index] = sum;
        }

        
        #endregion

    } // end class Program
} // end namespace MatrixMultiply
