/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_MAKEHERMITIAN_HPP
#define EL_MAKEHERMITIAN_HPP

namespace El {

template<typename T>
void MakeHermitian( UpperOrLower uplo, Matrix<T>& A );

template<typename T>
void MakeHermitian( UpperOrLower uplo, AbstractDistMatrix<T>& A );

} // namespace El

#endif // ifndef EL_MAKEHERMITIAN_HPP
