global using matrix_index = System.Int32;
global using matrix_data = System.Single;

using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;




namespace MatrixMultiply
{
    public static class Utils
    {
        /// <summary>
        /// Print a matrix to the console
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        public static void PrintMatrix(dynamic a)
        {
            for (matrix_index i = 0; i < a.GetLength(0); i++)
            {
                for (matrix_index j = 0; j < a.GetLength(1); j++)
                {
                    Console.Write(a[i,j] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        
        
        /// <summary>
        /// Creates a matrix populated with random values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated with random values</returns>
        [SuppressMessage(
            "Security",
            "CA5394:Do not use insecure randomness",
            Justification = "Only used for testing")]
        public static matrix_data[,] CreateRandomMatrix(matrix_index rows, matrix_index columns)
        {
            var rnd = new Random();
            var matrix = new matrix_data[rows, columns];

            for (matrix_index i = 0; i < rows; i++)
            {
                for (matrix_index j = 0; j < columns; j++)
                    matrix[i, j] = rnd.Next(minValue: -100, maxValue: 100);
            }

            return matrix;
        }


        /// <summary>
        /// Creates a matrix populated with sequential values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated with sequential values</returns>
        public static matrix_data[,] CreateSequentialMatrix(matrix_index rows, matrix_index columns)
        {
            matrix_data seq = 0.0f;
            var matrix = new matrix_data[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                    matrix[i, j] = seq++;
            }

            return matrix;
        }


        /// <summary>
        /// Creates a n-diagonal matrix populated with sequential values.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <param name="band_width">Band-width around the diagonal to populate</param>
        /// <returns>A banded diagonal matrix populated with sequential values</returns>
        public static matrix_data[,] CreateBandedSequentialMatrix(matrix_index rows, matrix_index columns, matrix_index band_width)
        {
            matrix_data seq = 0.0f;
            var matrix = new matrix_data[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++) 
                {
                    if(Math.Abs(i-j) < band_width) 
                    {
                        matrix[i, j] = seq++;
                    } else 
                    {
                        matrix[i, j] = 0.0f;
                    }
                }
                    
            }

            return matrix;
        }

        /// <summary>
        /// Creates an Identity matrix.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix</param>
        /// <param name="columns">The number of columns in the matrix</param>
        /// <returns>A matrix populated 1s along the diagonal and 0s elsewhere</returns>
        public static matrix_data[,] CreateIndentityMatrix(matrix_index rows, matrix_index columns)
        {
            var matrix = new matrix_data[rows, columns];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++) 
                {
                    if(i==j) 
                    {
                        matrix[i, j] = 1.0f;
                    }
                    else 
                    {
                        matrix[i, j] = 0.0f;
                    }
                }
                    
            }
            return matrix;
        }

        /// <summary>
        /// Compares two matrices for equality.
        /// </summary>
        /// <param name="a">A MxN matrix (the actual matrix we got)</param>
        /// <param name="e">A MxN matrix (the matrix we expected) </param>
        /// <returns>True if the matrices are equal</returns>
        public static bool MatrixEqual(dynamic a, dynamic e)
        {
            var ma = a.GetLength(0);
            var na = a.GetLength(1);
            var me = e.GetLength(0);
            var ne = e.GetLength(1);

            if (ma != me || na != ne)
            {
                Debug.WriteLine($"Matrix dimensions do not match: [{ma}x{na}] vs [{me}x{ne}]");
                return false;
            }

            for (var i = 0; i < ma; i++)
            {
                for (var j = 0; j < na; j++)
                {
                    var actual = a[i, j].ToString("G4");  // G4 = 4 significant digits
                    var expected = e[i, j].ToString("G4");
                    if (actual != expected) 
                    {
                        Debug.WriteLine($"Error at element location [{i}, {j}]: {actual} found, {expected} expected");
                        return false;
                    }
                }
            }

            return true;
        }


        /// <summary>
        /// Multiplies two dense matrices and returns the resultant matrix.
        /// </summary>
        /// <param name="accelerator">The Accelerator to run the multiplication on</param>
        /// <param name="a">A dense MxK matrix</param>
        /// <param name="b">A dense KxN matrix</param>
        /// <returns>A dense MxN matrix</returns>
        public static matrix_data[,] MatrixMultiplyNaive(dynamic a, dynamic b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

            var c = new matrix_data[m, n];

            for (var x = 0; x < m; x++)
            {
                for (var y = 0; y < n; y++)
                {
                    c[x, y] = 0;

                    for (var z = 0; z < ka; z++)
                        c[x, y] += a[x, z] * b[z, y];
                }
            }

            return c;
        }

        /// <summary>
        /// Compute the transpose of a matrix.
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <returns>The transpose of A, a NxM matrix</returns>
        public static matrix_data[,] MatrixTranspose(matrix_data[,] a)
        {
            matrix_index rows = (matrix_index) a.GetLength(0);
            matrix_index columns = (matrix_index) a.GetLength(1);
            var a_t = new matrix_data[columns, rows];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    a_t[j, i] = a[i, j];
                }
            }

            return a_t;
        }

        /// <summary>
        /// Apply a binary mask to a matrix
        /// </summary>
        /// <param name="a">A MxN matrix</param>
        /// <param name="mask">A MxN binary mask of 1s and 0s</param>
        /// <returns>(a && mask)  that is, the matrix a as masked by mask</returns>
        public static matrix_data[,] MatrixMask(matrix_data[,] a, matrix_data[,] mask)
        {
            var ma = a.GetLength(0);
            var na = a.GetLength(1);
            var mm = mask.GetLength(0);
            var nm = mask.GetLength(1);

            if (ma != mm || na != nm)
            {
                var err = $"Matrix dimensions do not match: [{ma}x{na}] vs [{mm}x{nm}]";
                Debug.WriteLine(err);
                throw new ArgumentException(err);
            }

            var a_m = new matrix_data[ma, na];

            for (var i = 0; i < ma; i++)
            {
                for (var j = 0; j < na; j++)
                {
                    if (mask[i, j] != 0.0f)
                    {
                        a_m[i, j] = a[i, j];
                    } 
                    else 
                    {
                        a_m[i, j] = 0.0f;
                    }  
                }
            }

            return a_m;
        }

        public static matrix_index[] To1DArray(matrix_index[,] input)
        {
            // Step 1: get total size of 2D array, and allocate 1D array.
            matrix_index size = (matrix_index) input.Length;
            matrix_index[] result = new matrix_index[size];
            
            // Step 2: copy 2D array elements into a 1D array.
            matrix_index write = 0;
            for (matrix_index i = 0; i <= input.GetUpperBound(0); i++)
            {
                for (matrix_index z = 0; z <= input.GetUpperBound(1); z++)
                {
                    result[write++] = input[i, z];
                }
            }
            // Step 3: return the new array.
            return result;
        }
    } // end class Utils   

    static class ArraySliceExt
    {
        public static ArraySlice2D<T> Slice<T>(this T[,] arr, matrix_index firstDimension, matrix_index length)
        {
            return new ArraySlice2D<T>(arr, firstDimension, length);
        }
    }
    class ArraySlice2D<T>
    {
        private readonly T[,] arr;
        private readonly matrix_index firstDimension;
        private readonly matrix_index length;
        public matrix_index Length { get { return length; } }
        public ArraySlice2D(T[,] arr, matrix_index firstDimension, matrix_index length)
        {
            this.arr = arr;
            this.firstDimension = firstDimension;
            this.length = length;
        }
        public T this[matrix_index index]
        {
            get { return arr[firstDimension, index]; }
            set { arr[firstDimension, index] = value; }
        }

        public matrix_index BinarySearch(T value)
        {
            matrix_index left = 0;
            matrix_index right = this.length - 1;
        
            matrix_index middle;
            while (left <= right)
            {
                middle = (left + right) / 2;
                switch (Compare(this[middle], value))
                {
                    case -1: left = middle + 1; break;
                    case 0: return middle;
                    case 1: right = middle - 1; break;
                }
            }
            throw new ArgumentOutOfRangeException("Element not found");
        }

        private int Compare(T x, T y)
        {
            return Comparer<T>.Default.Compare(x, y);
        }
    }
} // end namespace MatrixMultiply