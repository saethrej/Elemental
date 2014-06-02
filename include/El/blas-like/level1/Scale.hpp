/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_SCALE_HPP
#define EL_SCALE_HPP

namespace El {

template<typename T>
inline void
Scale( T alpha, Matrix<T>& A )
{
    DEBUG_ONLY(CallStackEntry cse("Scale"))
    if( alpha != T(1) )
    {
        if( alpha == T(0) )
            for( Int j=0; j<A.Width(); ++j )
                for( Int i=0; i<A.Height(); ++i )
                    A.Set(i,j,0);
        else
            for( Int j=0; j<A.Width(); ++j )
                blas::Scal( A.Height(), alpha, A.Buffer(0,j), 1 );
    }
}

template<typename Real>
inline void
Scale( Complex<Real> alpha, Matrix<Real>& AReal, Matrix<Real>& AImag )
{
    DEBUG_ONLY(CallStackEntry cse("Scale"))
    typedef Complex<Real> C;
    const Int m = AReal.Height();
    const Int n = AReal.Width();
    if( alpha != C(1) )
    {
        if( alpha == C(0) )
        {
            Scale( Real(0), AReal );
            Scale( Real(0), AImag );
        }
        else
        {
            Matrix<Real> aReal, aImag, aRealCopy, aImagCopy;
            for( Int j=0; j<n; ++j )
            {
                aReal = View( aReal, 0, j, m, 1 );
                aImag = View( aImag, 0, j, m, 1 );
                aRealCopy = aReal;
                aImagCopy = aImag;
                Scale( alpha.real(), aReal     );
                Axpy( -alpha.imag(), aImagCopy, aReal );
                Scale( alpha.real(), aImag     );
                Axpy(  alpha.imag(), aRealCopy, aImag );
            }
        }
    }
}

template<typename T>
inline void
Scale( Base<T> alpha, Matrix<T>& A )
{ Scale( T(alpha), A ); }

template<typename T>
inline void
Scale( T alpha, AbstractDistMatrix<T>& A )
{ Scale( alpha, A.Matrix() ); }

template<typename Real>
inline void
Scale
( Complex<Real> alpha, 
  AbstractDistMatrix<Real>& AReal, AbstractDistMatrix<Real>& AImag )
{ Scale( alpha, AReal.Matrix(), AImag.Matrix() ); }

template<typename T>
inline void
Scale( T alpha, AbstractBlockDistMatrix<T>& A )
{ Scale( alpha, A.Matrix() ); }

template<typename Real>
inline void
Scale
( Complex<Real> alpha, 
  AbstractBlockDistMatrix<Real>& AReal, AbstractBlockDistMatrix<Real>& AImag )
{ Scale( alpha, AReal.Matrix(), AImag.Matrix() ); }

template<typename T>
inline void
Scale( Base<T> alpha, AbstractDistMatrix<T>& A )
{ Scale( T(alpha), A.Matrix() ); }

template<typename T>
inline void
Scale( Base<T> alpha, AbstractBlockDistMatrix<T>& A )
{ Scale( T(alpha), A.Matrix() ); }

} // namespace El

#endif // ifndef EL_SCALE_HPP
