#ifndef HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERUTIL_HPP_
#define HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERUTIL_HPP_

#include <El/hydrogen_config.h>

#include <cusolverDn.h>

namespace hydrogen
{
namespace cusolver
{

/** @brief cuSOLVER uses ints to represent sizes. */
using SizeT = int;

/** @brief cuSOLVER uses ints to represent info variables. */
using InfoT = int;

/** @brief Convert a value to the size type expected by the cuSOLVER
 *         library.
 *
 *  If `HYDROGEN_DO_BOUNDS_CHECKING` is defined, this will do a
 *  "safe cast" (it will verify that `val` is in the dynamic range of
 *  `int`. Otherwise it will do a regular static_cast.
 */
template <typename T>
#ifdef HYDROGEN_DO_BOUNDS_CHECKING
SizeT ToSizeT(T const& val)
{
    return narrow_cast<SizeT>(val);
}
#else
SizeT ToSizeT(T const& val) noexcept
{
    return static_cast<SizeT>(val);
}
#endif // HYDROGEN_DO_BOUNDS_CHECKING

/** @brief Overload to prevent extra work in the case of dynamic range
 *         checking.
 */
inline SizeT ToSizeT(SizeT const& val) noexcept
{
    return val;
}

}// namespace cusolver
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_CUDA_CUSOLVERUTIL_HPP_
