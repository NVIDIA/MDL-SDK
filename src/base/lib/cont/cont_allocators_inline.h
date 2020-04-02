/******************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

 /// \file
 /// \brief  Implements allocators.
 ///
 ///         Allocators deal with the memory management and object's lifetime handling
 ///         and hence allow for encapsulating and modifying this in one place.
 ///         Possible allocators might include single-threaded, thread-safe and
 ///         pool-based allocators, for example.
 ///
 ///         All allocators have to conform to the interface
 ///
 ///            allocator_interface {
 ///             typedef T2_ Value_type;
 ///             typedef T3_ Pointer;
 ///             typedef T4_ Const_pointer;
 ///             typedef T5_ Reference;
 ///             typedef T6_ Const_reference;
 ///
 ///             void construct(Pointer, Const_reference);
 ///             void destroy(Pointer);
 ///             void construct_n(Pointer, Const_reference, size_t);
 ///             void destroy_n(Pointer, size_t);
 ///             Pointer allocate(size_t);
 ///             void deallocate(Pointer, size_t);
 ///
 ///             void uninitialized_copy(Pointer src, Pointer dest, size_t count);
 ///             void uninitialized_fill_n(Pointer, count, value);
 ///             void uninitialized_copy(Pointer start, Pointer end, Pointer dest);
 ///             void uninitialized_fill(Pointer, Pointer, value);
 ///            };

#include "i_cont_traits.h"
#include <base/lib/mem/mem.h>

#include <new>
#include <cstring>


namespace MI
{
namespace CONT
{

// Default constructor.
template <typename T>
inline
Default_allocator<T>::Default_allocator()
{}

// Copy constructor
template <typename T>
inline
Default_allocator<T>::Default_allocator(
    const Default_allocator&)		// the other allocator
{}

// Trivial destructor.
template <typename T>
inline
Default_allocator<T>::~Default_allocator()
{}


// Create an element at given position p with given value val.
template <typename T>
inline
void Default_allocator<T>::construct(
    Pointer p,				// pointer/place of element in memory
    Const_reference val)		// init value
{
    // init memory with placement new
    new(p) T(val);
}


// Destroy an element at the given position.
template <typename T>
inline
void Default_allocator<T>::destroy(
    Pointer p)				// pointer to element to destroy
{
    // calling p's destructor
    p->~T();
}


// Construct a range of objects.
// This is a convenience function based on single object construct().
template <typename T>
inline
void Default_allocator<T>::construct_n(
    Pointer p,				// starting pointer in memory
    Const_reference val,		// value for all elements
    size_t size)			// number of elements to be constructed
{
    // will be optimized soon
    for (size_t i=0; i<size; ++i, ++p)
	construct(p, val);
}


// Construct a range of objects.
template <typename T>
inline
void Default_allocator<T>::construct(
    Pointer start,			// place of first element in memory
    Pointer end,			// place of last element in memory
    Const_reference val)		// value for all elements
{
    // will be optimized soon
    for (; start != end; ++start)
	construct(start, val);
}


// Convenience function based on single object destroy().
//
// Destroy a range of objects. If the Value_type of the object has
// a trivial destructor, the compiler should optimize all of this
// away, otherwise the objects' destructors must be invoked.
template <typename T>
inline
void Default_allocator<T>::destroy_n(
    Pointer p,				// first element to be destroyed
    size_t size)			// number of elements to destroy
{
    typedef typename Type_traits<T>::has_trivial_destructor
	Has_trivial_destructor;
    // simply call the type-dependent optimized or default variant
    destroy_n_aux(p, size, Has_trivial_destructor());
}


// Convenience function based on single object destroy().
//
// Destroy a range of objects. If the Value_type of the object has
// a trivial destructor, the compiler should optimize all of this
// away, otherwise the objects' destructors must be invoked.
template <typename T>
inline
void Default_allocator<T>::destroy(
    Pointer start,			// place of first element in memory
    Pointer end)			// place of last element in memory
{
    typedef typename Type_traits<T>::has_trivial_destructor
	Has_trivial_destructor;
    // simply call the type-dependent optimized or default variant
    destroy_aux(start, end, Has_trivial_destructor());
}


// Allocate but don't initialize size elements of type T.
template <typename T>
inline
typename Default_allocator<T>::Pointer Default_allocator<T>::allocate(
    size_t size)			// number of elements
{
    // allocate memory with global new
#ifdef MI_MEM_TRACKER
#undef new
    return (MI::MEM::Context(__FILE__, __LINE__) * (Pointer) ::operator new(size*sizeof(T)));
#define new MI_MEM_TRACKER_NEW
#else // MI_MEM_TRACKER
    return (Pointer)(::operator new(size*sizeof(T)));
#endif // MI_MEM_TRACKER
}


// Deallocate storage p of deleted elements.
template <typename T>
inline
void Default_allocator<T>::deallocate(
    Pointer p,				// starting memory location
    size_t)				// dummy parameter
{
    // free memory with global delete
    ::operator delete((void*)p);
}


// Copy the given source memory of count objects into given raw memory.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy(
    Pointer src,			// source objects memory location
    Pointer target,			// destination objects memory location
    size_t count)			// number of elements to copy
{
    typedef typename Type_traits<T>::is_POD_type Is_POD;
    // simply call the type-dependent optimized or default variant
    return uninitialized_copy_aux(src, target, count, Is_POD());
}


// Copies the value count times into the range starting at target.
// This function does not require an initialized output range.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill_n(
    Pointer target,			// dest begin
    size_t count,			// number of copies to be made
    const T& value)			// value to copy
{
    typedef typename Type_traits<T>::is_POD_type Is_POD;
    // simply call the type-dependent optimized or default variant
    return uninitialized_fill_n_aux(target, count, value, Is_POD());
}


// Copy the given source memory range into given raw memory.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy(
    Pointer start,			// range begin
    Pointer end,			// range end
    Pointer target)			// destination objects memory location
{
    typedef typename Type_traits<T>::is_POD_type Is_POD;
    // simply call the type-dependent optimized or default variant
    return uninitialized_copy_aux(start, end, target, Is_POD());
}


// Copies the value count times into the range starting at target.
// This function does not require an initialized output range.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill(
    Pointer target,			// range begin
    Pointer end,			// range end
    const T& value)			// init value
{
    typedef typename Type_traits<T>::is_POD_type Is_POD;
    // simply call the type-dependent optimized or default variant
    return uninitialized_fill_aux(target, end, value, Is_POD());
}


//
//  Optimized type(traits) dependent implementations
//

// Destroy a range of objects with nontrivial destructors.
// This is a helper function used only by destroy().
template <typename T>
inline void Default_allocator<T>::destroy_n_aux(
    Pointer p,				// first element to destroy
    size_t count,			// number of elements to destroy
    const STLEXT::False_type&)			// type to differ
{
    for (size_t i=0; i<count; ++i)
	destroy(p++);
}


// Destroy a range of objects with trivial destructors.  Since the destructors
// are trivial, there's nothing to do and hopefully this function will be
// entirely optimized away.
template <typename T>
inline void Default_allocator<T>::destroy_n_aux(
    Pointer p,				// first element to destroy
    size_t count,			// number of elements to destroy
    const STLEXT::True_type&)			// type to differ
{}


// Destroy a range of objects with nontrivial destructors.
// This is a helper function used only by destroy().
template <typename T>
inline void Default_allocator<T>::destroy_aux(
    Pointer start,			// first element to destroy
    Pointer end,			// second element to destroy
    const STLEXT::False_type&)			// type to differ
{
    while (start != end)
	destroy(start++);
}


// Destroy a range of objects with trivial destructors.  Since the destructors
// are trivial, there's nothing to do and hopefully this function will be
// entirely optimized away.
template <typename T>
inline void Default_allocator<T>::destroy_aux(
    Pointer start,			// first element to destroy
    Pointer end,			// second element to destroy
    const STLEXT::True_type&)			// type to differ
{}


// Copy POD - no construction on the target site required. Hence we fall-
// back on memove (not memcopy due to overlapping).
template<typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy_aux(
    Pointer start,			// source begin
    Pointer end,			// source end
    Pointer target,			// destination objects memory location
    const STLEXT::True_type&)			// type to differ
{
    memmove(target, start, sizeof(T) * (end - start));
    return target + (end - start);
}


// Copy objects, which requires construction on the target size.
template<typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy_aux(
    Pointer start,			// source begin
    Pointer end,			// source end
    Pointer target,			// destination objects memory location
    const STLEXT::False_type&)			// type to differ
{
    Pointer current = target;
    for (; start != end; ++start, ++current)
	construct(current, *start);

    return current;
}


// Copy POD - no construction on the target site required. Hence we fall-
// back on memove (not memcopy due to overlapping).
template<typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy_aux(
    Pointer src,			// source begin
    Pointer target,			// destination objects memory location
    size_t count,			// number of elements to copy
    const STLEXT::True_type&)			// type to differ
{
    memmove(target, src, sizeof(T) * count);
    return target + count;
}


// Copy objects, which requires construction on the target size.
template<typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_copy_aux(
    Pointer src,			// source begin
    Pointer target,			// destination objects memory location
    size_t count,			// number of elements to copy
    const STLEXT::False_type&)			// type to differ
{
    Pointer current = target;

    for (size_t i=0; i<count; ++i, ++src, ++current)
	construct(current, *src);

    return current;
}


// Fill the given range with the given value.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill_aux(
    Pointer target,			// range begin
    Pointer end,			// range end
    const T& value,			// init value
    const STLEXT::False_type&)			// type to differ
{
    while (target != end)
	construct(target++, value);
    return target;
}


// Fill the given range with the given value.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill_aux(
    Pointer target,			// range begin
    Pointer end,			// range end
    const T& value,			// init value
    const STLEXT::True_type&)			// type to differ
{
    while (target != end)
	*target++ = value;
    return target;
}


// Copies the value count times into the range starting at target.
// This function does not require an initialized output range.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill_n_aux(
    Pointer target,			// target begin
    size_t count,			// number of copies to be made
    const T& value,			// init value
    const STLEXT::False_type&)			// type to differ
{
    for (size_t i=0; i<count; ++i)
	construct(target+i, value);
    return target + count;
}


// Copies the value count times into the range starting at target.
// This function does not require an initialized output range.
template <typename T>
inline
typename Default_allocator<T>::Pointer
Default_allocator<T>::uninitialized_fill_n_aux(
    Pointer target,			// target begin
    size_t count,			// number of copies to be made
    const T& value,			// init value
    const STLEXT::True_type&)			// type to differ
{
    for (size_t i=0; i<count; ++i)
	*(target+i) = value;
    return target + count;
}

} // namespace CONT
} // namespace MI
