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

#ifndef BASE_LIB_CONT_ALLOCATORS_H
#define BASE_LIB_CONT_ALLOCATORS_H

#include <cstddef>			// for size_t
#include <base/system/stlext/i_stlext_type_traits_base_types.h>

namespace MI {
namespace CONT {

// An implementation of an allocator - conforming to the allocator interface.
// This Default_allocator is based on the global operator new and delete
template <typename T>
class Default_allocator
{
  public:
    // Convenient typedefs - reused by the clients.
    typedef T			Value_type;
    typedef Value_type*		Pointer;
    typedef const Value_type*	Const_pointer;
    typedef Value_type&		Reference;
    typedef const Value_type&	Const_reference;

    // Alternate allocator type.
    template <typename U>
    struct rebind {
	typedef Default_allocator<U> other;
    };

    // Since this is a stateless allocator there is no real construction
    // code required - hence all constructors/destructors do nothing.
    // Default constructor.
    Default_allocator();

    // Copy constructor
    Default_allocator(
	const Default_allocator&);	// the other allocator

    // Conversion copy constructor
    template <typename U>
    Default_allocator(
	const Default_allocator<U>&)	// the other allocator
    {}

    // Trivial destructor.
    ~Default_allocator();


    // Initialize element at p with given value.
    void construct(
	Pointer p,			// place of element in memory
	Const_reference val);		// initial value of element
    // Destroy element at p.
    void destroy(
	Pointer p);			// place of element in memory

    // Construct a range of objects.
    void construct_n(
	Pointer p,			// place of first element in memory
	Const_reference val,		// initial value for all elements
	size_t size);			// number of elements
    // Destroy a range of objects.
    void destroy_n(
	Pointer p,			// place of first element in memory
	size_t size);			// number of elements
    // Construct a range of objects.
    void construct(
	Pointer start,			// place of first element in memory
	Pointer end,			// place of last element in memory
	Const_reference val);		// initial value for all elements
    // Destroy a range of objects.
    void destroy(
	Pointer start,			// place of first element in memory
	Pointer end);			// place of last element in memory

    // Allocate but don't initialize size elements of type T.
    Pointer allocate(
	size_t size);			// number of elements

    // Deallocate storage p of deleted elements.
    void deallocate(
	Pointer p,			// begin of storage to be freed
	size_t);			// dummy size of memory to free


    // Convenient functions
    //
    // Copy the given source memory of count objects into given raw memory.
    Pointer uninitialized_copy(
	Pointer src,			// source memory location
	Pointer target,			// target memory location
	size_t count);			// number of objects

    // Copies the value count times into the range starting at src.
    // This function does not require an initialized output range.
    Pointer uninitialized_fill_n(
	Pointer dest,			// destination begin
	size_t count,			// number of items to fill
	const T& value);		// init value

    // Copy the given source memory range into given raw memory.
    Pointer uninitialized_copy(
	Pointer start,			// source begin
	Pointer end,			// source end
	Pointer target);		// destination objects memory location

    // Copies the value count times into the range starting at src.
    // This function does not require an initialized output range.
    Pointer uninitialized_fill(
	Pointer start,			// range begin
	Pointer end,			// range end
	const T& value);		// init value


  private:
    // Type-dependent specializations for optimization.
    //
    // Destroy a range of objects with trivial destructors.
    void destroy_n_aux(
	Pointer p,			// first element to destroy
	size_t count,			// number of elements to destroy
	const STLEXT::True_type&);		// type to differ
    // Destroy a range of objects with nontrivial destructors.
    void destroy_n_aux(
	Pointer p,			// first element to destroy
	size_t count,			// number of elements to destroy
	const STLEXT::False_type&);		// type to differ
    // Destroy a range of objects with trivial destructors.
    void destroy_aux(
	Pointer start,			// first element to destroy
	Pointer end,			// end of range of objects
	const STLEXT::True_type&);		// type to differ
    // Destroy a range of objects with nontrivial destructors.
    void destroy_aux(
	Pointer start,			// first element to destroy
	Pointer end,			// end of range of objects
	const STLEXT::False_type&);		// type to differ
    // Copy POD, which do not require construction on the target size.
    Pointer uninitialized_copy_aux(
	Pointer start,			// source begin
	Pointer end,			// source end
	Pointer target,			// destination objects memory location
	const STLEXT::True_type&);		// type to differ
    // Copy objects, which requires construction on the target size.
    Pointer uninitialized_copy_aux(
	Pointer start,			// source begin
	Pointer end,			// source end
	Pointer target,			// destination objects memory location
	const STLEXT::False_type&);		// type to differ
    // Copy POD, which do not require construction on the target size.
    Pointer uninitialized_copy_aux(
	Pointer src,			// source begin
	Pointer target,			// destination objects memory location
	size_t count,			// number of elements to copy
	const STLEXT::True_type&);		// type to differ
    // Copy objects, which requires construction on the target size.
    Pointer uninitialized_copy_aux(
	Pointer src,			// source begin
	Pointer target,			// destination objects memory location
	size_t count,			// number of elements to copy
	const STLEXT::False_type&);		// type to differ
    // Fill the given range with the given value.
    Pointer uninitialized_fill_aux(
	Pointer start,			// range begin
	Pointer end,			// range end
	const T& value,			// init value
	const STLEXT::False_type&);		// type to differ
    // Fill the given range with the given value.
    Pointer uninitialized_fill_aux(
	Pointer start,			// range begin
	Pointer end,			// range end
	const T& value,			// init value
	const STLEXT::True_type&);		// type to differ
    // Copies the value count times into the range starting at dest.
    Pointer uninitialized_fill_n_aux(
	Pointer target,			// destination begin
	size_t count,			// number of copies to be made
	const T& value,			// init value
	const STLEXT::False_type&);		// type to differ
    // Copies the value count times into the range starting at dest.
    Pointer uninitialized_fill_n_aux(
	Pointer target,			// destination begin
	size_t count,			// number of copies to be made
	const T& value,			// init value
	const STLEXT::True_type&);		// type to differ

};

}
}

#include "cont_allocators_inline.h"

#endif
