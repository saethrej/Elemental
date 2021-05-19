/*
   Copyright (c) 2009-2016, Jack Poulson
                 2021, Jens Eirik Saethre
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;


/**
 * @brief runs the cholesky factorization with the desired parameters
 * 
 * @tparam F the 
 * @param g the grid to run the factorization on
 * @param uplo denotes whether to compute L or L^T
 * @param m the matrix dimension
 * @param nbLocal the local block size
 * @param scalapack flag indicating whether to test scalapack
 */
template<typename F>
void TestCholesky(const Grid& g, UpperOrLower uplo, Int m, Int nbLocal, bool scalapack)
{   
    OutputFromRoot(g.Comm(),"Testing distributed Cholesky with ",TypeName<F>());

    // initialize the matrix (currently to Identity)
    DistMatrix<F> A(g);
    SetLocalTrrkBlocksize<F>(nbLocal);
    Identity(A, m, m);

    // start factorization with timing
    mpi::Barrier(g.Comm());
    Timer timer;
    timer.Start();
    Cholesky(uplo, A, scalapack);
    mpi::Barrier( g.Comm() );
    const double runTime = timer.Stop() * 1000.0;

    // print timing
    OutputFromRoot(g.Comm(), runTime, " ms.");
}

int main(int argc, char* argv[])
{
    // set up the enviroment and the MPI communicator 
    Environment env( argc, argv );
    mpi::Comm comm = mpi::NewWorldComm();

    try {
        // parse input arguments
        Int gridHeight = Input("--gridHeight","process grid height",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const char uploChar = Input("--uplo","upper or lower storage: L/U",'L');
        const Int m = Input("--m","height of matrix",1024);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const Int nbLocal = Input("--nbLocal","local blocksize",32);
#ifdef EL_HAVE_SCALAPACK
        const bool scalapack = Input("--scalapack","test ScaLAPACK?",false);
#else
        const bool scalapack = false;
#endif
        // pass input arguments to the runtime enviroment and print details
        ProcessInput();
        PrintInputReport();

        // set up grid
        // default grid height if it was not specified (results in 2D)
        if (gridHeight == 0) {
            gridHeight = Grid::DefaultHeight(mpi::Size(comm));
        }
        const GridOrder order = colMajor ? COLUMN_MAJOR : ROW_MAJOR;
        const Grid g(std::move(comm), gridHeight, order);
        const UpperOrLower uplo = CharToUpperOrLower(uploChar);
        SetBlocksize(nb);

        // run cholesky factorization on all ranks
        TestCholesky<double>(g, uplo, m, nbLocal, scalapack);
    
    } catch(exception &e) {
        // report exception if one occured
        ReportException(e);
    }

    return 0; // success
}
