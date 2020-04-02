/***************************************************************************************************
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
 **************************************************************************************************/

 /// \file
 /// \brief Implements a dynamic array.
 ///
 ///        Arrays implement sequences of dynamically varying length.
 ///        Class array is parameterized with the type of the values.

#ifndef BASE_LIB_CONT_ARRAY_H
#define BASE_LIB_CONT_ARRAY_H

#include "cont_allocators.h"

namespace MI {
namespace CONT {

// A dynamically resizable array implementation. If used correctly, the
// underlying C array offers best caching behavior and fastest access
// and traversal.
template<typename T, typename Allocator=Default_allocator<T> >
class Array
{
  public:
    // Forward declarations
    class Iterator;
    class Const_iterator;

    // Required typedefs to allow different allocators.
    typedef typename Allocator::Pointer			Pointer;
    typedef typename Allocator::Const_pointer		Const_pointer;
    typedef typename Allocator::Reference		Reference;
    typedef typename Allocator::Const_reference		Const_reference;

    // STL-compliant typedefs
    typedef T						value_type;
    typedef Pointer					pointer;
    typedef Const_pointer				const_pointer;
    typedef Reference					reference;
    typedef Const_reference				const_reference;
    typedef Iterator					iterator;
    typedef Const_iterator				const_iterator;

    // Default constructor.
    explicit Array(
	const Allocator& alloc=Allocator()); 		// allocator

    // Constructor.
    explicit Array(
	size_t count,					// number of elements to create
	Const_reference value=T(),			// default value for those elements
	const Allocator& alloc=Allocator()); 		// allocator

    // Copy constructor.
    Array(
	const Array& array);				// array to copy

    // Destructor - free all resources.
    ~Array();

    // Assignment operator.
    Array& operator=(const Array& array); 		// array to assign

    // STL-style iterator access into the array. We use ordinary pointers as
    // iterators.
    Pointer	  begin();
    Const_pointer begin() const;
    Pointer	  end();
    Const_pointer end()	const;

    // Return number of elements in array.
    size_t size() const;

    // Return whether the array is empty or not.
    bool empty() const;

    // Reserve enough memory to hold the given number of elements.
    // Note that this function never shrinks memory, just increases it.
    void reserve(
	size_t count);					// number of elements

    // Resizes the vector to the specified number of elements.
    void resize(
	size_t new_size,				// no. of elements vector will contain
	Const_reference value=T());			// default value for new elements

    // Fast but unchecked access to element at index i.
    Reference operator[](
	size_t i);					// element index

    // Fast but unchecked access to element at index i.
    Const_reference operator[](
	size_t i) const;				// element index

    // Insert value at index. Return false if index is out of range, and
    // true otherwise.
    bool put(
	size_t index,					// index at which to put the value
	Const_reference value);				// value to insert

    // Put the given value at array[index]. If required, increase the size of
    // the array appropriately and initialize the new elements with default.
    void put(
	size_t index,					// index at which to put the value
	Const_reference value,				// value to insert
	Const_reference def_value);			// init value for new elements

    // Get element from array at index. Return false if index is out of range,
    // and true otherwise. Value is only set when return value is true.
    bool get(
	size_t index,					// index at which to put the value
	Pointer value) const;				// pointer for retrieve value

    // Get element from array at index. Return NULL if index is out of range,
    // and a pointer to the element otherwise.
    Pointer get(
	size_t index) const;				// index of element

    // Insert value at given index.
    // Note that this might imply shifting up all elements starting from index
    // and hence could result in run-time degradation. For inserting at the
    // end of the array the use of append() is preferable.
    void insert(
	size_t index,					// index at which to insert value
	Const_reference value);				// value to insert

    // Remove value at index, shifting down all elements above from index.
    // Return false if index  is out of range, and true otherwise.
    bool remove(
	size_t index);					// index of element to be removed

    // Insert value at the end of the array.
    // Note that this is a fast operation and should be the preferred way
    // to insert data into the array.
    void append(
	Const_reference value);				// value to insert

    // Insert a default constructed element at the end of the array.
    // Return a pointer to it. This is meant mainly for structures/classes
    // types which have no or very simple constructors but do the
    // initialization outside. They can be appended to the array without any
    // memory copy.
    Pointer append();

    // Note: The prepend() method is intentionally missing since it has such
    // a bad runtime complexity - exponentially, to be exact.

    // Clear the array, i.e. remove all elements.
    // Note that this function does not free any resources since they might
    // be reused. Freeing of resources happens inside the destructor.
    void clear();

    // Clear the array, i.e. remove all elements AND free the allocated memory.
    void clear_memory();

    // Fast data exchange of two Arrays.
    void swap(
	Array<T, Allocator>& other);			// the other

    // Iterator for traversing the elements.
    class Iterator
    {
    public:
	// Convenient typedef.
	typedef Array<T, Allocator> The_array;

	// Constructor.
	Iterator(
	    The_array& array);				// the array to iterate over
	// Destructor.
	~Iterator();

	// Set iterator to the first element of the array if it exists.
	void to_first();
	// Set iterator to the last element of the array if it exists.
	void to_last();
	// Set iterator to the next element of the array if it exists.
	void to_next();
	// Set iterator to the previous element of the array if it exists.
	void to_previous();

	// Return true if iterator is exhausted, false otherwise.
	bool at_end() const;

	// Return reference to the current element.
	typename The_array::Reference operator*();
	// Apply member selection to the current element.
	typename The_array::Pointer operator->();

    private:
	The_array& m_array;				// the corresponding array
	size_t m_current;				// index of current element in array
    };

    // The constant access iterator.
    class Const_iterator
    {
    public:
	// Convenient typedef.
	typedef Array<T, Allocator> The_array;

	// Constructor.
	Const_iterator(
	    const The_array& array);			// the array to iterate over
	// Destructor.
	~Const_iterator();

	// Set iterator to the first element of the array if it exists.
	void to_first();
	// Set iterator to the last element of the array if it exists.
	void to_last();
	// Set iterator to the next element of the array if it exists.
	void to_next();
	// Set iterator to the previous element of the array if it exists.
	void to_previous();

	// Return true if iterator is exhausted, false otherwise.
	bool at_end() const;

	// Return reference to the current element.
	typename The_array::Const_reference operator*() const;
	// Apply member selection to the current element.
	typename The_array::Const_pointer operator->() const;

    private:
	const The_array& m_array;			// the corresponding array
	size_t m_current;				// index of current element in array
    };

    // Return the internally allocated space in number of elements it can hold.
    size_t capacity() const;


  private:
    Allocator m_allocator;				// allocator in use
    Pointer   m_array;					// the data elements
    size_t m_count;					// number of elements
    size_t m_reserved;					// size of allocated memory

    // Resizes the vector to the specified number of elements.
    void do_resize(
	size_t new_size,				// no. of elements vector will contain
	Const_reference val);				// value for new elements

    // Insert value count times at given index.
    void insert_n(
	size_t index,					// index at which to insert value
	size_t count,					// how many times?
	Const_reference value);				// value to insert

    // Pure allocation function returning new memory fitting the given size.
    Pointer allocate(
	size_t new_count);				// no. of elements vector will contain
};


// Overload of the default swap() for Arrays.
// see Array::swap().
template<typename T, typename Allocator>
void swap(
    Array<T, Allocator>& one,				// the one
    Array<T, Allocator>& other);			// the other

}
}

#include "cont_array_inline.h"

#endif
