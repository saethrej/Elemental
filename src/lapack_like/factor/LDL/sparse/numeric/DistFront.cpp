/*
   Copyright 2009-2011, Jack Poulson.
   All rights reserved.

   Copyright 2011-2012, Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin.
   All rights reserved.

   Copyright 2013, Jack Poulson, Lexing Ying, and Stanford University.
   All rights reserved.

   Copyright 2013-2014, Jack Poulson and The Georgia Institute of Technology.
   All rights reserved.

   Copyright 2014-2015, Jack Poulson and Stanford University.
   All rights reserved.
   
   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {
namespace ldl {

template<typename F>
DistFront<F>::DistFront( DistFront<F>* parentNode )
: parent(parentNode), child(nullptr), duplicate(nullptr)
{ }

template<typename F>
DistFront<F>::DistFront
( const DistSparseMatrix<F>& A, 
  const DistMap& reordering,
  const DistSeparator& sep, 
  const DistNodeInfo& info,
  bool conjugate )
: parent(nullptr), child(nullptr), duplicate(nullptr)
{
    DEBUG_ONLY(CSE cse("DistFront::DistFront"))
    Pull( A, reordering, sep, info, conjugate );
}

// NOTE: 
// The current implementation (conjugate-)transposes A into the frontal tree
template<typename F>
void DistFront<F>::Pull
( const DistSparseMatrix<F>& A, 
  const DistMap& reordering,
  const DistSeparator& rootSep, 
  const DistNodeInfo& rootInfo,
  bool conjugate )
{
    DEBUG_ONLY(
      CSE cse("DistFront::Pull");
      if( A.LocalHeight() != reordering.NumLocalSources() )
          LogicError("Local mapping was not the right size");
    )
    const bool time = false;
   
    mpi::Comm comm = A.Comm();
    const int commSize = mpi::Size( comm );
    const int commRank = mpi::Rank( comm ); 
    Timer timer;

    // Compute the unique set of column indices that our process interacts with
    if( time && commRank == 0 )
        timer.Start();
    const Int* colBuffer = A.LockedTargetBuffer();
    const Int numLocalEntries = A.NumLocalEntries();
    vector<Int> colOffs(numLocalEntries);
    vector<ValueInt<Int>> uniqueCols(numLocalEntries);
    for( Int e=0; e<numLocalEntries; ++e )
        uniqueCols[e] = ValueInt<Int>{colBuffer[e],e};
    std::sort( uniqueCols.begin(), uniqueCols.end(), ValueInt<Int>::Lesser );
    {
        Int uniqueOff=-1, lastUnique=-1;
        for( Int e=0; e<numLocalEntries; ++e )
        {
            if( lastUnique != uniqueCols[e].value )
            {
                ++uniqueOff;
                lastUnique = uniqueCols[e].value;
                uniqueCols[uniqueOff] = uniqueCols[e];
            }
            colOffs[uniqueCols[e].index] = uniqueOff;
        }
        uniqueCols.resize( uniqueOff+1 );
    }
    const Int numUniqueCols = uniqueCols.size();
    if( time && commRank == 0 )
        Output("Unique sort: ",timer.Stop()," secs");

    // Get the reordered indices of our local rows of the sparse matrix
    if( time && commRank == 0 )
        timer.Start();
    const Int localHeightA = A.LocalHeight();
    vector<Int> mappedSources(localHeightA);
    for( Int iLoc=0; iLoc<localHeightA; ++iLoc ) 
        mappedSources[iLoc] = A.GlobalRow(iLoc);
    reordering.Translate( mappedSources );
    if( time && commRank == 0 )
        Output("Source translation: ",timer.Stop()," secs");

    // Get the reordered indices of the targets of our portion of the 
    // distributed sparse matrix
    if( time && commRank == 0 )
        timer.Start();
    vector<Int> mappedTargets(numUniqueCols);
    for( Int e=0; e<numUniqueCols; ++e )
        mappedTargets[e] = uniqueCols[e].value;
    reordering.Translate( mappedTargets );
    if( time && commRank == 0 )
        Output("Target translation: ",timer.Stop()," secs");

    // Set up the indices for the rows we need from each process
    if( time && commRank == 0 )
        timer.Start();
    vector<int> rRowSizes( commSize, 0 );
    function<void(const Separator&)> rRowLocalAccumulate = 
      [&]( const Separator& sep )
      {
          for( const Separator* child : sep.children )
             rRowLocalAccumulate( *child );
          for( const Int& i : sep.inds )
              ++rRowSizes[ A.RowOwner(i) ];
      };
    function<void(const DistSeparator&,const DistNodeInfo&)> 
      rRowAccumulate =
      [&]( const DistSeparator& sep, const DistNodeInfo& node )
      {
          if( sep.child == nullptr )
          {
              rRowLocalAccumulate( *sep.duplicate ); 
              return;
          }
          rRowAccumulate( *sep.child, *node.child );
          
          const Grid& grid = *node.grid;
          const Int rowShift = grid.Col();
          const Int rowStride = grid.Width();
          const Int numInds = sep.inds.size();
          for( Int t=rowShift; t<numInds; t+=rowStride )
              ++rRowSizes[ A.RowOwner(sep.inds[t]) ];
      };
    rRowAccumulate( rootSep, rootInfo );
    vector<int> rRowOffs;
    const Int numRecvRows = Scan( rRowSizes, rRowOffs );
    if( time && commRank == 0 )
        Output("Row index setup: ",timer.Stop()," secs");

    if( time && commRank == 0 )
        timer.Start();
    vector<Int> rRows( numRecvRows );
    auto offs = rRowOffs;
    function<void(const Separator&)> rRowsLocalPack = 
      [&]( const Separator& sep )
      {
          for( const Separator* childSep : sep.children )
              rRowsLocalPack( *childSep );
          for( Int i : sep.inds )
              rRows[offs[A.RowOwner(i)]++] = i;
      };
    function<void(const DistSeparator&,const DistNodeInfo&)> rRowsPack = 
      [&]( const DistSeparator& sep, const DistNodeInfo& node )
      {
          if( sep.child == nullptr )
          {
              rRowsLocalPack( *sep.duplicate );
              return;
          }
          rRowsPack( *sep.child, *node.child );
          
          const Grid& grid = *node.grid;
          const Int rowShift = grid.Col();
          const Int rowStride = grid.Width();
          const Int numInds = sep.inds.size();
          for( Int t=rowShift; t<numInds; t+=rowStride )
          {
              const Int i = sep.inds[t];
              rRows[offs[A.RowOwner(i)]++] = i;
          }
      };
    rRowsPack( rootSep, rootInfo );
    if( time && commRank == 0 )
        Output("Row index pack: ",timer.Stop()," secs");

    // Retreive the list of rows that we must send to each process
    if( time && commRank == 0 )
        timer.Start();
    vector<int> sRowSizes( commSize );
    mpi::AllToAll( rRowSizes.data(), 1, sRowSizes.data(), 1, comm );
    vector<int> sRowOffs;
    const Int numSendRows = Scan( sRowSizes, sRowOffs );
    vector<Int> sRows( numSendRows );
    mpi::AllToAll
    ( rRows.data(), rRowSizes.data(), rRowOffs.data(),
      sRows.data(), sRowSizes.data(), sRowOffs.data(), comm );
    if( time && commRank == 0 )
        Output("AllToAll: ",timer.Stop()," secs");

    // Pack the number of nonzeros per row (and the nonzeros themselves)
    // TODO: Avoid sending upper-triangular data
    if( time && commRank == 0 )
        timer.Start();
    const Int firstLocalRow = A.FirstLocalRow();
    vector<Int> sRowLengths( numSendRows );
    vector<int> sEntriesSizes(commSize,0);
    for( Int q=0; q<commSize; ++q )
    {
        const Int size = sRowSizes[q];
        const Int off = sRowOffs[q];
        for( Int s=0; s<size; ++s )
        {
            const Int i = sRows[s+off];
            const Int iLoc = i-firstLocalRow;
            const Int jReord = mappedSources[iLoc];
            const Int rowOff = A.EntryOffset( iLoc );
            const Int numConnections = A.NumConnections( iLoc );
            sRowLengths[s+off] = 0;
            for( Int e=0; e<numConnections; ++e )
            {
                const Int iReord = mappedTargets[colOffs[rowOff+e]];
                if( iReord >= jReord )
                {
                    ++sEntriesSizes[q];
                    ++sRowLengths[s+off];
                }
            }
        }
    }
    vector<int> sEntriesOffs;
    const int numSendEntries = Scan( sEntriesSizes, sEntriesOffs );
    vector<F> sEntries( numSendEntries );
    vector<Int> sTargets( numSendEntries );
    for( Int q=0; q<commSize; ++q )
    {
        Int index = sEntriesOffs[q];
        const Int size = sRowSizes[q];
        const Int off = sRowOffs[q];
        for( Int s=0; s<size; ++s )
        {
            const Int i = sRows[s+off];
            const Int iLoc = i-firstLocalRow;
            const Int jReord = mappedSources[iLoc];
            const Int rowOff = A.EntryOffset( iLoc );
            const Int numConnections = A.NumConnections( iLoc );
            for( Int e=0; e<numConnections; ++e )
            {
                const Int iReord = mappedTargets[colOffs[rowOff+e]];
                if( iReord >= jReord )
                {
                    const F value = A.Value( rowOff+e );
                    sEntries[index] = (conjugate ? Conj(value) : value);
                    sTargets[index] = iReord;
                    ++index;
                }
            }
        }
        DEBUG_ONLY(
          if( index != sEntriesOffs[q]+sEntriesSizes[q] )
              LogicError("index was not the correct value");
        )
    }
    if( time && commRank == 0 )
        Output("Payload pack: ",timer.Stop()," secs");

    // Send back the number of nonzeros per row and the nonzeros themselves
    if( time && commRank == 0 )
        timer.Start();
    vector<Int> rRowLengths( numRecvRows );
    mpi::AllToAll
    ( sRowLengths.data(), sRowSizes.data(), sRowOffs.data(),
      rRowLengths.data(), rRowSizes.data(), rRowOffs.data(), comm );
    vector<int> rEntriesSizes(commSize,0);
    for( Int q=0; q<commSize; ++q )
    {
        const Int size = rRowSizes[q];
        const Int off = rRowOffs[q];
        for( Int s=0; s<size; ++s )
            rEntriesSizes[q] += rRowLengths[off+s];
    }
    vector<int> rEntriesOffs;
    const int numRecvEntries = Scan( rEntriesSizes, rEntriesOffs );
    vector<F> rEntries( numRecvEntries );
    vector<Int> rTargets( numRecvEntries );
    mpi::AllToAll
    ( sEntries.data(), sEntriesSizes.data(), sEntriesOffs.data(),
      rEntries.data(), rEntriesSizes.data(), rEntriesOffs.data(), comm );
    mpi::AllToAll
    ( sTargets.data(), sEntriesSizes.data(), sEntriesOffs.data(),
      rTargets.data(), rEntriesSizes.data(), rEntriesOffs.data(), comm );
    if( time && commRank == 0 )
        Output("AllToAll time: ",timer.Stop()," secs");

    // Unpack the received entries
    if( time && commRank == 0 )
        timer.Start();
    offs = rRowOffs;
    auto entryOffs = rEntriesOffs;
    function<void(const Separator&,const NodeInfo&,Front<F>&)> 
      unpackEntriesLocal = 
      [&]( const Separator& sep, const NodeInfo& node, Front<F>& front )
      {
          front.type = SYMM_2D;
          front.isHermitian = conjugate;
 
          const Int numChildren = sep.children.size();
          front.children.resize( numChildren );
          for( Int c=0; c<numChildren; ++c )
          {
              front.children[c] = new Front<F>(&front);
              unpackEntriesLocal
              ( *sep.children[c], *node.children[c], *front.children[c] );
          }

          const Int size = node.size;
          const Int off = node.off;
          const Int lowerSize = node.lowerStruct.size();
          Zeros( front.L, size+lowerSize, size );

          for( Int t=0; t<size; ++t )
          {
              const Int i = sep.inds[t];
              const Int q = A.RowOwner(i);

              int& entryOff = entryOffs[q];
              const Int numEntries = rRowLengths[offs[q]++];

              for( Int k=0; k<numEntries; ++k )
              {
                  const F value = rEntries[entryOff];
                  const Int target = rTargets[entryOff];
                  ++entryOff;
  
                  if( target < off+t )
                      continue;
                  else if( target < off+size )
                  {
                      front.L.Set( target-off, t, value );
                  }
                  else
                  {
                      // TODO: Avoid this binary search?
                      const Int origOff = Find( node.origLowerStruct, target );
                      const Int row = node.origLowerRelInds[origOff];
                      front.L.Set( row, t, value );
                  }
              }
          }
      };
    function<void(const DistSeparator&,
                  const DistNodeInfo&,
                        DistFront<F>&)> unpackEntries = 
      [&]( const DistSeparator& sep, const DistNodeInfo& node, 
                 DistFront<F>& front )
      {
          front.type = SYMM_2D;
          front.isHermitian = conjugate;
          const Grid& grid = *node.grid;

          if( sep.child == nullptr )
          {
              front.duplicate = new Front<F>(&front);
              unpackEntriesLocal
              ( *sep.duplicate, *node.duplicate, *front.duplicate );

              front.L2D.Attach( grid, front.duplicate->L );

              return;
          }
          front.child = new DistFront<F>(&front);
          unpackEntries( *sep.child, *node.child, *front.child );

          const Int size = node.size;
          const Int off = node.off;
          const Int lowerSize = node.lowerStruct.size();
          front.L2D.SetGrid( grid );
          Zeros( front.L2D, size+lowerSize, size );
          
          const Int localWidth = front.L2D.LocalWidth();
          for( Int tLoc=0; tLoc<localWidth; ++tLoc )
          {
              const Int t = front.L2D.GlobalCol(tLoc);
              const Int i = sep.inds[t];
              const Int q = A.RowOwner(i);

              int& entryOff = entryOffs[q];
              const Int numEntries = rRowLengths[offs[q]++];

              for( Int k=0; k<numEntries; ++k )
              {
                  const F value = rEntries[entryOff];
                  const Int target = rTargets[entryOff];
                  ++entryOff;

                  if( target < off+t )
                      continue;
                  else if( target < off+size )
                  {
                      front.L2D.Set( target-off, t, value );
                  }
                  else 
                  {
                      // TODO: Avoid this binary search?
                      const Int origOff = Find( node.origLowerStruct, target );
                      const Int row = node.origLowerRelInds[origOff];
                      front.L2D.Set( row, t, value );
                  }
              }
          }
      };
    unpackEntries( rootSep, rootInfo, *this );
    if( time && commRank == 0 )
        Output("Unpack: ",timer.Stop()," secs");
    DEBUG_ONLY(
      for( Int q=0; q<commSize; ++q )
          if( entryOffs[q] != rEntriesOffs[q]+rEntriesSizes[q] )
              LogicError("entryOffs were incorrect");
    )
}

// TODO: Use faster approach used in Pull
template<typename F>
void DistFront<F>::PullUpdate
( const DistSparseMatrix<F>& A, 
  const DistMap& reordering,
  const DistSeparator& rootSep, 
  const DistNodeInfo& rootInfo )
{
    DEBUG_ONLY(
      CSE cse("DistFront::PullUpdate");
      if( A.LocalHeight() != reordering.NumLocalSources() )
          LogicError("Local mapping was not the right size");
    )
    const bool time = false;
   
    mpi::Comm comm = A.Comm();
    const DistGraph& graph = A.LockedDistGraph();
    const int commSize = mpi::Size( comm );
    const int commRank = mpi::Rank( comm ); 
    Timer timer;

    // Get the reordered indices of the targets of our portion of the 
    // distributed sparse matrix
    if( time && commRank == 0 )
        timer.Start();
    set<Int> targetSet( graph.targets_.begin(), graph.targets_.end() );
    vector<Int> targets;
    CopySTL( targetSet, targets );
    auto mappedTargets = targets;
    reordering.Translate( mappedTargets );
    if( time && commRank == 0 )
        Output("Translation: ",timer.Stop()," secs");

    // Set up the indices for the rows we need from each process
    if( time && commRank == 0 )
        timer.Start();
    vector<int> rRowSizes( commSize, 0 );
    function<void(const Separator&)> rRowLocalAccumulate = 
      [&]( const Separator& sep )
      {
          for( const Separator* child : sep.children )
             rRowLocalAccumulate( *child );
          for( const Int& i : sep.inds )
              ++rRowSizes[ A.RowOwner(i) ];
      };
    function<void(const DistSeparator&,const DistNodeInfo&)> 
      rRowAccumulate =
      [&]( const DistSeparator& sep, const DistNodeInfo& node )
      {
          if( sep.child == nullptr )
          {
              rRowLocalAccumulate( *sep.duplicate ); 
              return;
          }
          rRowAccumulate( *sep.child, *node.child );
          
          const Grid& grid = *node.grid;
          const Int rowShift = grid.Col();
          const Int rowStride = grid.Width();
          const Int numInds = sep.inds.size();
          for( Int t=rowShift; t<numInds; t+=rowStride )
              ++rRowSizes[ A.RowOwner(sep.inds[t]) ];
      };
    rRowAccumulate( rootSep, rootInfo );
    vector<int> rRowOffs;
    const Int numRecvRows = Scan( rRowSizes, rRowOffs );
    if( time && commRank == 0 )
        Output("Row index setup: ",timer.Stop()," secs");

    if( time && commRank == 0 )
        timer.Start();
    vector<Int> rRows( numRecvRows );
    auto offs = rRowOffs;
    function<void(const Separator&)> rRowsLocalPack = 
      [&]( const Separator& sep )
      {
          for( const Separator* childSep : sep.children )
              rRowsLocalPack( *childSep );
          for( Int i : sep.inds )
              rRows[offs[A.RowOwner(i)]++] = i;
      };
    function<void(const DistSeparator&,const DistNodeInfo&)> rRowsPack = 
      [&]( const DistSeparator& sep, const DistNodeInfo& node )
      {
          if( sep.child == nullptr )
          {
              rRowsLocalPack( *sep.duplicate );
              return;
          }
          rRowsPack( *sep.child, *node.child );
          
          const Grid& grid = *node.grid;
          const Int rowShift = grid.Col();
          const Int rowStride = grid.Width();
          const Int numInds = sep.inds.size();
          for( Int t=rowShift; t<numInds; t+=rowStride )
          {
              const Int i = sep.inds[t];
              rRows[offs[A.RowOwner(i)]++] = i;
          }
      };
    rRowsPack( rootSep, rootInfo );
    if( time && commRank == 0 )
        Output("Row index pack: ",timer.Stop()," secs");

    // Retreive the list of rows that we must send to each process
    if( time && commRank == 0 )
        timer.Start();
    vector<int> sRowSizes( commSize );
    mpi::AllToAll( rRowSizes.data(), 1, sRowSizes.data(), 1, comm );
    vector<int> sRowOffs;
    const Int numSendRows = Scan( sRowSizes, sRowOffs );
    vector<Int> sRows( numSendRows );
    mpi::AllToAll
    ( rRows.data(), rRowSizes.data(), rRowOffs.data(),
      sRows.data(), sRowSizes.data(), sRowOffs.data(), comm );
    if( time && commRank == 0 )
        Output("AllToAll: ",timer.Stop()," secs");

    // Pack the number of nonzeros per row (and the nonzeros themselves)
    // TODO: Avoid sending upper-triangular data
    if( time && commRank == 0 )
        timer.Start();
    const Int firstLocalRow = A.FirstLocalRow();
    vector<Int> sRowLengths( numSendRows );
    vector<int> sEntriesSizes(commSize,0);
    for( Int q=0; q<commSize; ++q )
    {
        const Int size = sRowSizes[q];
        const Int off = sRowOffs[q];
        for( Int s=0; s<size; ++s )
        {
            const Int i = sRows[s+off];
            const Int numConnections = A.NumConnections( i-firstLocalRow );
            sEntriesSizes[q] += numConnections;
            sRowLengths[s+off] = numConnections;
        }
    }
    vector<int> sEntriesOffs;
    const int numSendEntries = Scan( sEntriesSizes, sEntriesOffs );
    vector<F> sEntries( numSendEntries );
    vector<Int> sTargets( numSendEntries );
    for( Int q=0; q<commSize; ++q )
    {
        Int index = sEntriesOffs[q];
        const Int size = sRowSizes[q];
        const Int off = sRowOffs[q];
        for( Int s=0; s<size; ++s )
        {
            const Int i = sRows[s+off];
            const Int numConnections = sRowLengths[s+off];
            const Int localEntryOff = A.EntryOffset( i-firstLocalRow );
            for( Int t=0; t<numConnections; ++t )
            {
                const F value = A.Value( localEntryOff+t );
                const Int col = A.Col( localEntryOff+t );
                const Int targetOff = Find( targets, col );
                const Int mappedTarget = mappedTargets[targetOff];
                sEntries[index] = (this->isHermitian ? Conj(value) : value);
                sTargets[index] = mappedTarget;
                ++index;
            }
        }
        DEBUG_ONLY(
          if( index != sEntriesOffs[q]+sEntriesSizes[q] )
              LogicError("index was not the correct value");
        )
    }
    if( time && commRank == 0 )
        Output("Payload pack: ",timer.Stop()," secs");

    // Send back the number of nonzeros per row and the nonzeros themselves
    if( time && commRank == 0 )
        timer.Start();
    vector<Int> rRowLengths( numRecvRows );
    mpi::AllToAll
    ( sRowLengths.data(), sRowSizes.data(), sRowOffs.data(),
      rRowLengths.data(), rRowSizes.data(), rRowOffs.data(), comm );
    vector<int> rEntriesSizes(commSize,0);
    for( Int q=0; q<commSize; ++q )
    {
        const Int size = rRowSizes[q];
        const Int off = rRowOffs[q];
        for( Int s=0; s<size; ++s )
            rEntriesSizes[q] += rRowLengths[off+s];
    }
    vector<int> rEntriesOffs;
    const int numRecvEntries = Scan( rEntriesSizes, rEntriesOffs );
    vector<F> rEntries( numRecvEntries );
    vector<Int> rTargets( numRecvEntries );
    mpi::AllToAll
    ( sEntries.data(), sEntriesSizes.data(), sEntriesOffs.data(),
      rEntries.data(), rEntriesSizes.data(), rEntriesOffs.data(), comm );
    mpi::AllToAll
    ( sTargets.data(), sEntriesSizes.data(), sEntriesOffs.data(),
      rTargets.data(), rEntriesSizes.data(), rEntriesOffs.data(), comm );
    if( time && commRank == 0 )
        Output("AllToAll time: ",timer.Stop()," secs");

    // Unpack the received updates
    if( time && commRank == 0 )
        timer.Start();
    offs = rRowOffs;
    auto entryOffs = rEntriesOffs;
    function<void(const Separator&,const NodeInfo&,Front<F>&)> 
      unpackEntriesLocal = 
      [&]( const Separator& sep, const NodeInfo& node, Front<F>& front )
      {
          const Int numChildren = sep.children.size();
          for( Int c=0; c<numChildren; ++c )
          {
              unpackEntriesLocal
              ( *sep.children[c], *node.children[c], *front.children[c] );
          }

          const Int size = node.size;
          const Int off = node.off;
          const Int lowerSize = node.lowerStruct.size();
          for( Int t=0; t<size; ++t )
          {
              const Int i = sep.inds[t];
              const Int q = A.RowOwner(i);

              int& entryOff = entryOffs[q];
              const Int numEntries = rRowLengths[offs[q]++];

              for( Int k=0; k<numEntries; ++k )
              {
                  const F value = rEntries[entryOff];
                  const Int target = rTargets[entryOff];
                  ++entryOff;
  
                  if( target < off+t )
                      continue;
                  else if( target < off+size )
                  {
                      front.L.Update( target-off, t, value );
                  }
                  else
                  {
                      const Int origOff = Find( node.origLowerStruct, target );
                      const Int row = node.origLowerRelInds[origOff];
                      front.L.Update( row, t, value );
                  }
              }
          }
      };
    function<void(const DistSeparator&,
                  const DistNodeInfo&,
                        DistFront<F>&)> unpackEntries = 
      [&]( const DistSeparator& sep, const DistNodeInfo& node, 
                 DistFront<F>& front )
      {
          const Grid& grid = *node.grid;

          if( sep.child == nullptr )
          {
              unpackEntriesLocal
              ( *sep.duplicate, *node.duplicate, *front.duplicate );
              return;
          }
          unpackEntries( *sep.child, *node.child, *front.child );

          const Int size = node.size;
          const Int off = node.off;
          const Int lowerSize = node.lowerStruct.size();
          const Int localWidth = front.L2D.LocalWidth();
          for( Int tLoc=0; tLoc<localWidth; ++tLoc )
          {
              const Int t = front.L2D.GlobalCol(tLoc);
              const Int i = sep.inds[t];
              const Int q = A.RowOwner(i);

              int& entryOff = entryOffs[q];
              const Int numEntries = rRowLengths[offs[q]++];

              for( Int k=0; k<numEntries; ++k )
              {
                  const F value = rEntries[entryOff];
                  const Int target = rTargets[entryOff];
                  ++entryOff;

                  if( target < off+t )
                      continue;
                  else if( target < off+size )
                  {
                      front.L2D.Update( target-off, t, value );
                  }
                  else 
                  {
                      const Int origOff = Find( node.origLowerStruct, target );
                      const Int row = node.origLowerRelInds[origOff];
                      front.L2D.Update( row, t, value );
                  }
              }
          }
      };
    unpackEntries( rootSep, rootInfo, *this );
    if( time && commRank == 0 )
        Output("Unpack: ",timer.Stop()," secs");
    DEBUG_ONLY(
      for( Int q=0; q<commSize; ++q )
          if( entryOffs[q] != rEntriesOffs[q]+rEntriesSizes[q] )
              LogicError("entryOffs were incorrect");
    )
}

template<typename F>
void DistFront<F>::Push
( DistSparseMatrix<F>& A, 
  const DistMap& reordering,
  const DistSeparator& rootSep, 
  const DistNodeInfo& rootInfo ) const
{
    DEBUG_ONLY(CSE cse("DistFront::Push"))
    LogicError("This routine needs to be written");
}

template<typename F>
void DistFront<F>::Unpack
( DistSparseMatrix<F>& A, 
  const DistSeparator& rootSep, 
  const DistNodeInfo& rootInfo ) const
{
    DEBUG_ONLY(CSE cse("DistFront::Unpack"))
    mpi::Comm comm = rootInfo.grid->Comm();
    const int commSize = mpi::Size(comm);
    A.SetComm( comm );
    const Int n = rootInfo.off + rootInfo.size;
    Zeros( A, n, n );
    
    // Compute the metadata for sending the entries of the frontal tree
    // ================================================================
    // TODO: Avoid metadata redundancy within each row of a front
    vector<int> sendSizes(commSize,0);
    function<void(const Separator&,
                  const NodeInfo&,
                  const Front<F>&)> localCount =
      [&]( const Separator& sep, const NodeInfo& node, 
           const Front<F>& front )
      {
        const Int numChildren = sep.children.size();
        for( Int c=0; c<numChildren; ++c )
            localCount
            ( *sep.children[c], *node.children[c], *front.children[c] );

        for( Int s=0; s<node.size; ++s )
        {
            const int q = A.RowOwner(node.off+s);
            for( Int t=0; t<=s; ++t ) 
                sendSizes[q]++;
        }

        const Int structSize = node.lowerStruct.size();
        for( Int s=0; s<structSize; ++s ) 
        {
            const int q = A.RowOwner(node.lowerStruct[s]);
            for( Int t=0; t<node.size; ++t )
                sendSizes[q]++;
        }
      };
    function<void(const DistSeparator&,
                  const DistNodeInfo&,
                  const DistFront<F>&)> count =  
      [&]( const DistSeparator& sep, const DistNodeInfo& node,
           const DistFront<F>& front )
      {
        if( sep.duplicate != nullptr )
        {
            localCount( *sep.duplicate, *node.duplicate, *front.duplicate );
            return;
        }
        count( *sep.child, *node.child, *front.child );

        if( FrontIs1D(front.type) )
        {
            const Int frontHeight = front.L1D.Height();
            auto FTL = front.L1D( IR(0,node.size), IR(0,node.size) );
            auto FBL = front.L1D( IR(node.size,frontHeight), IR(0,node.size) );
            
            const Int localWidth = FTL.LocalWidth(); 
            const Int topLocalHeight = FTL.LocalHeight();
            const Int botLocalHeight = FBL.LocalHeight();

            for( Int sLoc=0; sLoc<topLocalHeight; ++sLoc )
            {
                const Int s = FTL.GlobalRow(sLoc);
                const Int i = node.off + s;
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                    if( FTL.GlobalCol(tLoc) <= s )
                        ++sendSizes[q];
            }

            for( Int sLoc=0; sLoc<botLocalHeight; ++sLoc )
            {
                const Int s = FBL.GlobalRow(sLoc);
                const Int i = node.lowerStruct[s];
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                    ++sendSizes[q];
            }
        }
        else
        {
            const Int frontHeight = front.L2D.Height();
            auto FTL = front.L2D( IR(0,node.size), IR(0,node.size) );
            auto FBL = front.L2D( IR(node.size,frontHeight), IR(0,node.size) );

            const Int localWidth = FTL.LocalWidth(); 
            const Int topLocalHeight = FTL.LocalHeight();
            const Int botLocalHeight = FBL.LocalHeight();

            for( Int sLoc=0; sLoc<topLocalHeight; ++sLoc )
            {
                const Int s = FTL.GlobalRow(sLoc);
                const int q = A.RowOwner(node.off+s);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                    if( FTL.GlobalCol(tLoc) <= s )
                        ++sendSizes[q];
            }

            for( Int sLoc=0; sLoc<botLocalHeight; ++sLoc )
            {
                const Int s = FBL.GlobalRow(sLoc);
                const Int i = node.lowerStruct[s];
                const int q = A.RowOwner(i);
                for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                    ++sendSizes[q];
            }
        }
      };
    count( rootSep, rootInfo, *this );

    // Pack the data
    // =============
    vector<int> sendOffs;
    const int totalSend = Scan( sendSizes, sendOffs );
    vector<Entry<F>> sendBuf(totalSend);
    auto offs = sendOffs;
    function<void(const Separator&,
                  const NodeInfo&,
                  const Front<F>&)> localPack =
      [&]( const Separator& sep, const NodeInfo& node, 
           const Front<F>& front )
      {
        const Int numChildren = sep.children.size();
        for( Int c=0; c<numChildren; ++c )
            localPack
            ( *sep.children[c], *node.children[c], *front.children[c] );

        for( Int s=0; s<node.size; ++s )
        {
            const Int i = node.off + s;
            const int q = A.RowOwner(i);
            for( Int t=0; t<=s; ++t ) 
            {
                F value = front.L.Get(s,t);
                sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
            }
        }

        const Int structSize = node.lowerStruct.size();
        for( Int s=0; s<structSize; ++s ) 
        {
            const Int i = node.lowerStruct[s];
            const int q = A.RowOwner(i);
            for( Int t=0; t<node.size; ++t )
            {
                F value = front.L.Get(node.size+s,t);
                sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
            }
        }
      };
    function<void(const DistSeparator&,
                  const DistNodeInfo&,
                  const DistFront<F>&)> pack =  
      [&]( const DistSeparator& sep, const DistNodeInfo& node,
           const DistFront<F>& front )
      {
        if( sep.duplicate != nullptr )
        {
            localPack( *sep.duplicate, *node.duplicate, *front.duplicate );
            return;
        }
        pack( *sep.child, *node.child, *front.child );

        if( FrontIs1D(front.type) )
        {
            const Int frontHeight = front.L1D.Height();
            auto FTL = front.L1D( IR(0,node.size), IR(0,node.size) );
            auto FBL = front.L1D( IR(node.size,frontHeight), IR(0,node.size) );
            
            const Int localWidth = FTL.LocalWidth(); 
            const Int topLocalHeight = FTL.LocalHeight();
            const Int botLocalHeight = FBL.LocalHeight();

            for( Int sLoc=0; sLoc<topLocalHeight; ++sLoc )
            {
                const Int s = FTL.GlobalRow(sLoc);
                const Int i = node.off + s;
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                {
                    const Int t = FTL.GlobalCol(tLoc);
                    if( t <= s )
                    {
                        F value = FTL.GetLocal(sLoc,tLoc);
                        sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
                    }
                }
            }

            for( Int sLoc=0; sLoc<botLocalHeight; ++sLoc )
            {
                const Int s = FBL.GlobalRow(sLoc);
                const Int i = node.lowerStruct[s];
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                {
                    Int t = FBL.GlobalCol(tLoc);
                    F value = FBL.GetLocal(sLoc,tLoc);
                    sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
                }
            }
        }
        else
        {
            const Int frontHeight = front.L2D.Height();
            auto FTL = front.L2D( IR(0,node.size), IR(0,node.size) );
            auto FBL = front.L2D( IR(node.size,frontHeight), IR(0,node.size) );

            const Int localWidth = FTL.LocalWidth(); 
            const Int topLocalHeight = FTL.LocalHeight();
            const Int botLocalHeight = FBL.LocalHeight();

            for( Int sLoc=0; sLoc<topLocalHeight; ++sLoc )
            {
                const Int s = FTL.GlobalRow(sLoc);
                const Int i = node.off + s;
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                {
                    const Int t = FTL.GlobalCol(tLoc);
                    if( t <= s )
                    {
                        F value = FTL.GetLocal(sLoc,tLoc);
                        sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
                    }
                }
            }

            for( Int sLoc=0; sLoc<botLocalHeight; ++sLoc )
            {
                const Int s = FBL.GlobalRow(sLoc);
                const Int i = node.lowerStruct[s];
                const int q = A.RowOwner(i);
                for( Int tLoc=0; tLoc<localWidth; ++tLoc )
                {
                    Int t = FBL.GlobalCol(tLoc);
                    F value = FBL.GetLocal(sLoc,tLoc);
                    sendBuf[offs[q]++] = Entry<F>{ i, node.off+t, value };
                }
            }
        }
      };
    pack( rootSep, rootInfo, *this );

    // Exchange and unpack
    // ===================
    auto recvBuf = mpi::AllToAll( sendBuf, sendSizes, sendOffs, comm );
    A.Reserve( recvBuf.size() );
    for( auto& entry : recvBuf )
        A.QueueUpdate( entry );
    A.ProcessQueues();
}

template<typename F>
const DistFront<F>& 
DistFront<F>::operator=( const DistFront<F>& front )
{
    DEBUG_ONLY(CSE cse("DistFront::operator="))
    isHermitian = front.isHermitian;
    type = front.type;
    if( front.child == nullptr )
    {
        child = nullptr;
        duplicate = new Front<F>(this);
        *duplicate = *front.duplicate;
        const Grid& grid = front.L2D.Grid();
        L2D.Attach( grid, front.duplicate->L );
        diag.Attach( grid, front.duplicate->diag );
        subdiag.Attach( grid, front.duplicate->subdiag );
        piv.Attach( grid, front.duplicate->piv );
    }
    else
    {
        duplicate = nullptr;
        child = new DistFront<F>(this);  
        *child = *front.child;
        L1D = front.L1D;
        L2D = front.L2D;
        diag = front.diag;
        subdiag = front.subdiag;
        piv = front.piv;
        work = front.work;
    }
    return *this;
}

template<typename F>
Int DistFront<F>::NumLocalEntries() const
{
    DEBUG_ONLY(CSE cse("DistFront::NumLocalEntries"))
    Int numEntries = 0;
    function<void(const DistFront<F>&)> count =
      [&]( const DistFront<F>& front )
      {
        if( front.duplicate != nullptr )
        {
            numEntries += front.duplicate->NumEntries();
            return;
        }
        count( *front.child );

        // Add in L
        numEntries += front.L1D.LocalHeight() * front.L1D.LocalWidth();
        numEntries += front.L2D.LocalHeight() * front.L2D.LocalWidth();
        
        // Add in the workspace
        numEntries += front.work.LocalHeight() * front.work.LocalWidth();
      };
    count( *this );
    return numEntries;
}

template<typename F>
Int DistFront<F>::NumTopLeftLocalEntries() const
{
    DEBUG_ONLY(CSE cse("DistFront::NumTopLeftLocalEntries"))
    Int numEntries = 0;
    function<void(const DistFront<F>&)> count =
      [&]( const DistFront<F>& front )
      {
        if( front.duplicate != nullptr )
        {
            numEntries += front.duplicate->NumTopLeftEntries();
            return;
        }
        count( *front.child );

        if( FrontIs1D(front.type) ) 
        {
            const Int n = front.L1D.Width();
            auto FTL = front.L1D( IR(0,n), IR(0,n) );
            numEntries += FTL.LocalHeight() * FTL.LocalWidth();
        }
        else
        {
            const Int n = front.L2D.Width();
            auto FTL = front.L2D( IR(0,n), IR(0,n) );
            numEntries += FTL.LocalHeight() * FTL.LocalWidth();
        }
      };
    count( *this );
    return numEntries;
}

template<typename F>
Int DistFront<F>::NumBottomLeftLocalEntries() const
{
    DEBUG_ONLY(CSE cse("DistFront::NumBottomLeftLocalEntries"))
    Int numEntries = 0;
    function<void(const DistFront<F>&)> count =
      [&]( const DistFront<F>& front )
      {
        if( front.duplicate != nullptr )
        {
            numEntries += front.duplicate->NumBottomLeftEntries();
            return;
        }
        count( *front.child );

        if( FrontIs1D(front.type) ) 
        {
            const Int m = front.L1D.Height();
            const Int n = front.L1D.Width();
            auto FBL = front.L2D( IR(n,m), IR(0,n) );
            numEntries += FBL.LocalHeight() * FBL.LocalWidth();
        }
        else
        {
            const Int m = front.L2D.Height();
            const Int n = front.L2D.Width();
            auto FBL = front.L2D( IR(n,m), IR(0,n) );
            numEntries += FBL.LocalHeight() * FBL.LocalWidth();
        }
      };
    count( *this );
    return numEntries;
}

template<typename F>
double DistFront<F>::LocalFactorGFlops( bool selInv ) const
{
    DEBUG_ONLY(CSE cse("DistFront::LocalFactorGFlops"))
    double gflops = 0.;
    function<void(const DistFront<F>&)> count =
      [&]( const DistFront<F>& front )
      {
        if( front.duplicate != nullptr )
        {
            gflops += front.duplicate->FactorGFlops();
            return;
        }
        count( *front.child );

        double m, n, p;
        if( FrontIs1D(front.type) ) 
        {
            m = front.L1D.Height();
            n = front.L1D.Width();
            p = front.L1D.DistSize();
        }
        else
        {
            m = front.L2D.Height();
            n = front.L2D.Width();
            p = front.L2D.DistSize();
        }
        double realFrontFlops = 
          ( selInv ? (2*n*n*n/3) + (m-n)*n + (m-n)*(m-n)*n
                   : (1*n*n*n/3) + (m-n)*n + (m-n)*(m-n)*n ) / p;
        gflops += (IsComplex<F>::val ? 4*realFrontFlops : realFrontFlops)/1.e9;
      };
    count( *this );
    return gflops;
}

template<typename F>
double DistFront<F>::LocalSolveGFlops( Int numRHS ) const
{
    DEBUG_ONLY(CSE cse("DistFront::LocalSolveGFlops"))
    double gflops = 0.;
    function<void(const DistFront<F>&)> count =
      [&]( const DistFront<F>& front )
      {
        if( front.duplicate != nullptr )
        {
            gflops += front.duplicate->SolveGFlops( numRHS );
            return;
        }
        count( *front.child );

        double m, n, p;
        if( FrontIs1D(front.type) ) 
        {
            m = front.L1D.Height();
            n = front.L1D.Width();
            p = front.L1D.DistSize();
        }
        else
        {
            m = front.L2D.Height();
            n = front.L2D.Width();
            p = front.L2D.DistSize();
        }
        double realFrontFlops = (m*n*numRHS) / p;
        gflops += (IsComplex<F>::val ? 4*realFrontFlops : realFrontFlops)/1.e9;
      };
    count( *this );
    return gflops;
}

template<typename F>
void DistFront<F>::ComputeRecvInds( const DistNodeInfo& info ) const
{
    DEBUG_ONLY(CSE cse("DistFront::ComputeRecvInds"))

    vector<int> gridHeights, gridWidths;
    GetChildGridDims( info, gridHeights, gridWidths );

    const int teamSize = mpi::Size( info.comm );
    const int teamRank = mpi::Rank( info.comm );
    const bool onLeft = info.child->onLeft;
    const int childTeamSize = mpi::Size( info.child->comm );
    const int childTeamRank = mpi::Rank( info.child->comm );
    const bool inFirstTeam = ( childTeamRank == teamRank );
    const bool leftIsFirst = ( onLeft==inFirstTeam );
    vector<int> teamSizes(2), teamOffs(2);
    teamSizes[0] = ( onLeft ? childTeamSize : teamSize-childTeamSize );
    teamSizes[1] = teamSize - teamSizes[0];
    teamOffs[0] = ( leftIsFirst ? 0            : teamSizes[1] );
    teamOffs[1] = ( leftIsFirst ? teamSizes[0] : 0            );
    DEBUG_ONLY(
      if( teamSizes[0] != gridHeights[0]*gridWidths[0] )
          RuntimeError("Computed left grid incorrectly");
      if( teamSizes[1] != gridHeights[1]*gridWidths[1] )
          RuntimeError("Computed right grid incorrectly");
    )

    commMeta.childRecvInds.resize( teamSize );
    for( int q=0; q<teamSize; ++q )
        commMeta.childRecvInds[q].clear();
    for( Int c=0; c<2; ++c )
    {
        // Compute the recv indices of the child from each process 
        const Int numInds = info.childRelInds[c].size();
        vector<Int> rowInds, colInds;
        for( Int iChild=0; iChild<numInds; ++iChild )
        {
            if( L2D.IsLocalRow( info.childRelInds[c][iChild] ) )
                rowInds.push_back( iChild );
            if( L2D.IsLocalCol( info.childRelInds[c][iChild] ) )
                colInds.push_back( iChild );
        }

        vector<Int>::const_iterator it;
        const Int numColInds = colInds.size();
        const Int numRowInds = rowInds.size();
        for( Int jPre=0; jPre<numColInds; ++jPre )
        {
            const Int jChild = colInds[jPre];
            const Int j = info.childRelInds[c][jChild];
            const Int jLoc = L2D.LocalCol(j);

            const int childCol = (jChild+info.childSizes[c]) % gridWidths[c];

            // Find the first iPre that maps to the lower triangle
            it = std::lower_bound( rowInds.begin(), rowInds.end(), jChild );
            const Int iPreStart = Int(it-rowInds.begin());
            for( Int iPre=iPreStart; iPre<numRowInds; ++iPre )
            {
                const Int iChild = rowInds[iPre];
                const Int i = info.childRelInds[c][iChild];
                DEBUG_ONLY(
                  if( iChild < jChild )
                      LogicError("Invalid iChild");
                )
                const Int iLoc = L2D.LocalRow(i);

                const int childRow = (iChild+info.childSizes[c])%gridHeights[c];
                const int childRank = childRow + childCol*gridHeights[c];

                const int q = teamOffs[c]+ childRank;
                commMeta.childRecvInds[q].push_back(iLoc);
                commMeta.childRecvInds[q].push_back(jLoc);
            }
        }
    }
}

template<typename F>
void DistFront<F>::ComputeCommMeta
( const DistNodeInfo& info, bool computeRecvInds ) const
{
    DEBUG_ONLY(CSE cse("DistFront::ComputeCommMeta"))
    commMeta.Empty();
    if( child == nullptr )
        return;

    const int teamSize = L2D.DistSize();
    commMeta.numChildSendInds.resize( teamSize );
    El::MemZero( commMeta.numChildSendInds.data(), teamSize );
    
    auto& childFront = *child;
    auto& childInfo = *info.child;
    const auto& childRelInds =
      ( childInfo.onLeft ? info.childRelInds[0] : info.childRelInds[1] );

    const auto& FBR = childFront.work;
    const Int localHeight = FBR.LocalHeight();
    const Int localWidth = FBR.LocalWidth();
    for( Int jChildLoc=0; jChildLoc<localWidth; ++jChildLoc )
    {
        const Int jChild = FBR.GlobalCol(jChildLoc);
        const int j = childRelInds[jChild];
        for( Int iChildLoc=0; iChildLoc<localHeight; ++iChildLoc )
        {
            const Int iChild = FBR.GlobalRow(iChildLoc);
            if( iChild >= jChild )
            {
                const Int i = childRelInds[iChild];
                const int q = L2D.Owner( i, j );
                ++commMeta.numChildSendInds[q];
            }
        }
    }

    // This is optional since it requires a nontrivial amount of storage.
    if( computeRecvInds )
        ComputeRecvInds( info );
}

#define PROTO(F) template struct DistFront<F>;
#define EL_NO_INT_PROTO
#include "El/macros/Instantiate.h"

} // namespace ldl
} // namespace El
