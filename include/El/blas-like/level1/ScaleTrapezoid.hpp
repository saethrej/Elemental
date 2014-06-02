/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_SCALETRAPEZOID_HPP
#define EL_SCALETRAPEZOID_HPP

namespace El {

template<typename T>
inline void
ScaleTrapezoid( T alpha, UpperOrLower uplo, Matrix<T>& A, Int offset=0 )
{
    DEBUG_ONLY(CallStackEntry cse("ScaleTrapezoid"))
    const Int height = A.Height();
    const Int width = A.Width();
    const Int ldim = A.LDim();
    T* buffer = A.Buffer();

    if( uplo == UPPER )
    {
        EL_PARALLEL_FOR
        for( Int j=Max(0,offset-1); j<width; ++j )
        {
            const Int numRows = j-offset+1;
            for( Int i=0; i<numRows; ++i )
                buffer[i+j*ldim] *= alpha;
        }
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            const Int numZeroRows = Max(j-offset,0);
            for( Int i=numZeroRows; i<height; ++i )
                buffer[i+j*ldim] *= alpha;
        }
    }
}

template<typename T>
inline void
ScaleTrapezoid
( T alpha, UpperOrLower uplo, AbstractDistMatrix<T>& A, Int offset=0 )
{
    DEBUG_ONLY(CallStackEntry cse("ScaleTrapezoid"))
    const Int height = A.Height();
    const Int localHeight = A.LocalHeight();
    const Int localWidth = A.LocalWidth();

    if( uplo == UPPER )
    {
        T* buffer = A.Buffer();
        const Int ldim = A.LDim();
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            const Int lastRow = j-offset;
            const Int boundary = Min( lastRow+1, height );
            const Int numRows = A.LocalRowOffset(boundary);
            T* col = &buffer[jLoc*ldim];
            for( Int iLoc=0; iLoc<numRows; ++iLoc )
                col[iLoc] *= alpha;
        }
    }
    else
    {
        T* buffer = A.Buffer();
        const Int ldim = A.LDim();
        EL_PARALLEL_FOR
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            const Int firstRow = Max(j-offset,0);
            const Int numZeroRows = A.LocalRowOffset(firstRow);
            T* col = &buffer[numZeroRows+jLoc*ldim];
            for( Int iLoc=0; iLoc<(localHeight-numZeroRows); ++iLoc )
                col[iLoc] *= alpha;
        }
    }
}

} // namespace El

#endif // ifndef EL_SCALETRAPEZOID_HPP
