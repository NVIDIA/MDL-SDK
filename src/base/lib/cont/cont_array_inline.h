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
 /// \brief  Implements a dynamic array.
 ///
 ///        Arrays implement sequences of dynamically varying length.
 ///        Class array is parametrized with the type of the values.

#include <base/lib/log/i_log_assert.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <base/system/main/i_module_id.h>
#include <algorithm>

namespace MI {
namespace CONT {

//=============================================================================

//-----------------------------------------------------------------------------

// Return the number of reserved space.
// Note: This is implemented on top of all other functions since the
// destructor is already using this function.
template<typename T, typename Allocator>
inline
size_t Array<T, Allocator>::capacity() const
{
    return m_reserved;
}


//-----------------------------------------------------------------------------

// Constructor.
template<typename T, typename Allocator>
inline
Array<T, Allocator>::Array(
    const Allocator& alloc)		// allocator
  : m_allocator(alloc), m_array(0)
{
    m_reserved = m_count = 0;
}


//-----------------------------------------------------------------------------

// Constructor.
template<typename T, typename Allocator>
inline
Array<T, Allocator>::Array(
    size_t count,			// number of elements
    Const_reference val,		// initial value of (all) elements
    const Allocator& alloc)		// allocator
  : m_allocator(alloc), m_array(0)
{
    if (count == 0) {
        m_reserved = m_count = 0;
    }
    else {
        // allocate memory
        m_array = m_allocator.allocate(count);
        m_reserved = m_count = count;

        // initialize elements in place
        m_allocator.construct_n(m_array, val, count);
    }
    ASSERT(M_CONT, m_count <= m_reserved);
}

//-----------------------------------------------------------------------------

// Copy constructor.
template<typename T, typename Allocator>
inline
Array<T, Allocator>::Array(
    const Array<T, Allocator>& array)	// the array to copy from
  : m_allocator(array.m_allocator), m_array(0)
{
    m_reserved = m_count = array.size();

    // copy memory from array to this
    if (array.size()) {
        // allocate memory
        m_array = m_allocator.allocate(array.size());
        // copy memory
        T* p_dest = m_array;
        T* p_src = array.m_array;
        while (p_dest < m_array + array.size())
            m_allocator.construct(p_dest++, *p_src++);
    }
}


//-----------------------------------------------------------------------------

// Destructor.
template<typename T, typename Allocator>
inline
Array<T, Allocator>::~Array()
{
    // clear data
    clear();

    // clear memory
    m_allocator.deallocate(m_array, capacity());
    m_array = 0;
    m_reserved = 0;
}


//-----------------------------------------------------------------------------

// Assignment operator.
template<typename T, typename Allocator>
inline
Array<T, Allocator>& Array<T, Allocator>::operator=(
    const Array<T, Allocator>& array)	// the array to assign from
{
    // check for self assignment
    if (this == &array)
        return *this;

    // use same allocator
    m_allocator = array.m_allocator;

    // clean-up old memory
    clear();
    // is there sufficiently enough memory?
    if (capacity() < array.size()) {
        m_allocator.deallocate(m_array, m_count);
        m_array = m_allocator.allocate(array.size());
        m_reserved = array.size();
    }
    // else re-use memory

    // copy data from array to this
    T* p_dest = m_array;
    T* p_src  = array.m_array;
    while (p_dest < m_array + array.size())
        m_allocator.construct(p_dest++, *p_src++);
    // alternatively use this:
    //for (size_t i=0; i<array.size(); ++i, ++p)
    //	m_allocator.construct(p, array[i]);

    m_count = array.size();

    return *this;
}


//-----------------------------------------------------------------------------

// STL-style iterator access

template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer Array<T, Allocator>::begin()
{
    return m_array;
}

template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Const_pointer Array<T, Allocator>::begin() const
{
    return m_array;
}

template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer Array<T, Allocator>::end()
{
    return begin() + size();
}

template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Const_pointer Array<T, Allocator>::end() const
{
    return begin() + size();
}


//-----------------------------------------------------------------------------

// Return number of elements in array.
template<typename T, typename Allocator>
inline
size_t Array<T, Allocator>::size() const
{
    return m_count;
}


//-----------------------------------------------------------------------------

// Return whether the array is empty pr not.
template<typename T, typename Allocator>
inline
bool Array<T, Allocator>::empty() const
{
    return size() == 0;
}


//-----------------------------------------------------------------------------

// Reserve enough memory to hold the given number of elements.
// Note that this function never shrinks memory, just encreases it.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::reserve(
    size_t size)			// number of elements
{
    typedef Array<T, Allocator> This;

    // reserve never shrinks memory
    if (size <= capacity())
        return;

    /// allocate new memory
    Pointer tmp = m_allocator.allocate(size);

    // move and free old memory
    if (m_array) {
        m_allocator.uninitialized_copy(m_array, tmp, This::size());
        m_allocator.destroy_n(m_array, This::size());
        m_allocator.deallocate(m_array, capacity());
    }

    // reset values
    m_array = tmp;
    m_reserved = size;
    //m_count has not changed.
}


//-----------------------------------------------------------------------------

// This function will resize the vector to the specified number of elements.
// If the number is smaller than the vector's current size the vector is
// truncated, otherwise the vector is extended and new elements are left
// uninitialized.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::resize(
    size_t new_size,			// no. of elements vector will contain
    Const_reference value)		// default value for new elements
{
    do_resize(new_size, value);
}


//-----------------------------------------------------------------------------

// Convenient but unsafe access to element at index i.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Reference Array<T, Allocator>::operator[](
    size_t i)				// index of element
{
    ASSERT(M_CONT, i < size());
    return m_array[i];
}


//-----------------------------------------------------------------------------

// Convenient but unsafe access to element at index i.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Const_reference Array<T, Allocator>::operator[](
    size_t i) const			// index of element
{
    ASSERT(M_CONT, i < size());
    return m_array[i];
}


//-----------------------------------------------------------------------------

// Insert value at index. Return false if index is out of range, else true.
template<typename T, typename Allocator>
inline
bool Array<T, Allocator>::put(
    size_t index,			// index at which to put the value
    Const_reference value)		// value to insert
{
    if (index >= size())
        return false;

    m_array[index] = value;
    return true;
}


//-----------------------------------------------------------------------------

// Put the given value at array[index]. If required, encrease the size of
// the array appropriately and initialize the new elements with default.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::put(
    size_t index,			// index at which to put the value
    Const_reference value,		// value to insert
    Const_reference def_value)		// init value for new elements
{
    // insertion w/o increasing size
    if (put(index, value))
        return;
// TO DO:
    // insertion with increasing size
    do_resize(index+1, def_value);
    m_array[index] = value;
}


//-----------------------------------------------------------------------------

// Get element from array at index. Return false if index is out of range,
// and true otherwise. Value is only set when return value is true.
template<typename T, typename Allocator>
inline
bool Array<T, Allocator>::get(
    size_t index,			// index at which to put the value
    Pointer value) const		// pointer for retrieve value
{
    if (index >= size())
        return false;

    *value = m_array[index];
    return true;
}


//-----------------------------------------------------------------------------

// Get element from array at index. Return NULL if index is out of range,
// and a pointer to the element otherwise.
template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer Array<T, Allocator>::get(
    size_t index) const			// index of element to retrieve
{
    if (index >= size())
        return 0;

    return &m_array[index];
}


//-----------------------------------------------------------------------------

// Insert value at given index.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::insert(
    size_t index,			// index at which to insert value
    Const_reference value)		// value to insert
{
    if (index < size())
        insert_n(index, 1, value);
    else {
        // insert index default values and than append the given value
        reserve(index+1);
        resize(index);
        append(value);
    }
}


//-----------------------------------------------------------------------------

// Insert value count times at given index.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::insert_n(
    size_t index,			// index at which to insert value
    size_t count,			// how many times?
    Const_reference value)		// value to insert
{
    // since this is an internal routine which should be fast
    // --> handle exceptional cases externally
    ASSERT(M_CONT, index < size());

    if (count == 0)
        return;

    T tmp = value;			// avoid problems if value is in array

    // resize array accordingly
    size_t old_count = size();
    size_t new_count = index < size()? size()+count : index+count;

    // ie, need to reallocate
    if (capacity() < new_count) {
        Pointer new_array = allocate(new_count);

        // copy first part of old array
        Pointer ptr = m_allocator.uninitialized_copy(
            m_array, new_array, index);

        // insert count values
        ptr = m_allocator.uninitialized_fill_n(ptr, count, tmp);

        // copy last part of old array
        ptr = m_allocator.uninitialized_copy(
            m_array+index, ptr, old_count-index);

        ASSERT(M_CONT, ptr == new_array+new_count);

        // free old memory and assign ptr to m_array
        m_allocator.destroy_n(m_array, old_count);
        m_allocator.deallocate(m_array, capacity());

        m_array = new_array;

        m_reserved = new_count;
        m_count = new_count;
    }
    else {
        resize(new_count);

        // move data one up iff required
        if (index < old_count) {
            // move memory count times up, starting at m_array+index: backwards
            T* p_old = m_array+old_count-1;	// last old element
            T* p_new = m_array+size()-1;	// last new position
            ASSERT(M_CONT, p_old+1 == p_new);	// ???? <-- don't understand it

// TO DO: uninitialized_copy + anschliessend destroy_n --> Optimierung
// shoud become something like copy_backward()
#ifdef DEBUG
            size_t counter = 0;
            T* start = p_old;
#endif
            while (p_old > m_array+index) {
                m_allocator.construct(p_new--, *p_old);
                m_allocator.destroy(p_old--);
#ifdef DEBUG
                ++counter;
#endif
            }
#ifdef DEBUG
            ASSERT(M_CONT, p_new == start + count);
#endif
        }

        // insert count values
        m_allocator.uninitialized_fill_n(m_array+index, count, tmp);
    }
}


//-----------------------------------------------------------------------------

// Remove value at index, shifting down all elements above from index.
// Return false if index is out of range, and true otherwise.
template<typename T, typename Allocator>
inline
bool Array<T, Allocator>::remove(
    size_t index)			// index of element to be removed
{
    if (index >= size())
        return false;

    // call element's destructor
    m_allocator.destroy(&m_array[index]);

    if (index < size()-1) {
        // move elements right from index one index down to the left
        T* p_old = m_array+index+1;
        T* p_new = m_array+index;

// TO_DO uninitialized_copy + anschliessend destroy_n --> Optimierung
        //memmove(m_array+index, m_array+index+1, sizeof(T)*(size()-index-1));
        while (p_old < m_array+size()) {
            m_allocator.construct(p_new++, *p_old);
            m_allocator.destroy(p_old++);
        }
    }

    --m_count;
    return true;
}


//-----------------------------------------------------------------------------

// Insert value at the end of the array.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::append(
    Const_reference value)		// value to insert
{
    if (m_count >= capacity()) {
        T tmp = value;			// avoid problems if value is in array

        size_t new_size = (3*capacity())/2;
        if (m_count >= new_size)
            new_size = m_count + 1;
        reserve(new_size);
        m_allocator.construct(m_array + m_count, tmp);
    }
    else
        // here we don't need the temp copy since the memory was not moved!
        m_allocator.construct(m_array + m_count, value);

    ++m_count;
}


//-----------------------------------------------------------------------------

// Insert a default constructed element at the end of the array.
// Return a pointer to it. This is meant mainly for structures/classes types
// which have no or very simple constructors but do the initialization outside.
// They can be appended to the array without any memory copy.
template<typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer
    Array<T, Allocator>::append()
{
    size_t i = m_count;
    if (m_count >= capacity()) {
        size_t new_size = (3*capacity())/2;
        if (m_count >= new_size)
            new_size = m_count + 1;
        reserve(new_size);
    }
    ++m_count;
    // Make sure the constructor - if existing - is called for the new element
    new(&m_array[i]) T();
    return &m_array[i];
}


//-----------------------------------------------------------------------------

// Clear array - ie destroy all elements.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::clear()
{
    // call destructors for all elements
    m_allocator.destroy_n(m_array, m_count);

    // memory is intentionally NOT freed - it might be reused instead
    m_count = 0;
}


//-----------------------------------------------------------------------------

// Clear array - ie destroy all elements AND free allocated memory.
template<typename T, typename Allocator>
inline
void Array<T, Allocator>::clear_memory()
{
    clear();

    // clear memory
    m_allocator.deallocate(m_array, capacity());
    m_array = 0;
    m_reserved = 0;
}


//-----------------------------------------------------------------------------

// Swaps this array another array.
// This function swaps two arrays by exchanging the elements, which is done
// in constant time. Note that the global swap() functionality falls back to
// this function due to its template specialization.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::swap(
    Array<T, Allocator>& other)		// the other
{
    using std::swap;
    swap(m_allocator, other.m_allocator);
    swap(m_array, other.m_array);
    swap(m_count, other.m_count);
    swap(m_reserved, other.m_reserved);
}


//-----------------------------------------------------------------------------

// This function will resize the vector to the specified number of elements.
// If the number is smaller than the vector's current size the vector is
// truncated, otherwise the vector is extended and new elements are populated
// with given data.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::do_resize(
    size_t new_count,			// no. of elements vector will contain
    Const_reference val)		// data for new elements
{
    if (new_count < size()) {
        // shorten array from the right by size()-new_count elements
        m_allocator.destroy_n(&m_array[new_count], size()-new_count);
    }
    else {
        // do we have to reallocate memory?
        if (new_count > capacity()) {
            T tmp_val = val;		// avoid problems if value is in array

            // create enough space for current data + new one
            size_t new_size = (3*capacity())/2;
            if (new_count > new_size)
                new_size = new_count;

            // allocate memory for final data
            T* tmp = m_allocator.allocate(new_size);

            // move and free old memory
            if (m_array) {
                m_allocator.uninitialized_copy(m_array, tmp, size());
                m_allocator.destroy_n(m_array, size());
                m_allocator.deallocate(m_array, capacity());
            }

            // assign tmp to m_array
            m_array = tmp;
            m_reserved = new_size;

            // now allocate elements from size() upto new_count
            m_allocator.construct_n(m_array+size(), tmp_val, new_count-size());
        }
        else
            // now allocate elements from size() upto new_count
            m_allocator.construct_n(m_array+size(), val, new_count-size());
    }

    m_count = new_count;
}


//-----------------------------------------------------------------------------

// The pure allocation function returning new memory fitting the given size.
// Note: internal routine which assumes new_count > capacity!!!
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer
Array<T, Allocator>::allocate(
    size_t new_count)			// no. of elements vector will contain
{
    // do we have to reallocate memory?
    ASSERT(M_CONT, new_count > capacity());

    // create enough space for current data + new one
    size_t new_size = (3*capacity())/2;
    if (new_count > new_size)
        new_size = new_count;

    // allocate memory for final data
    return m_allocator.allocate(new_size);
}


//=============================================================================

//-----------------------------------------------------------------------------

// Constructor.
template <typename T, typename Allocator>
inline
Array<T, Allocator>::Iterator::Iterator(
    Array<T, Allocator>& array)		// The array to iterate over.
  : m_array(array), m_current(0)
{}


//-----------------------------------------------------------------------------

// Destructor.
template <typename T, typename Allocator>
inline
Array<T, Allocator>::Iterator::~Iterator()
{}


//-----------------------------------------------------------------------------

// Set iterator to the first element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Iterator::to_first()
{
    m_current = 0;
}


//-----------------------------------------------------------------------------

// Set iterator to the last element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Iterator::to_last()
{
    m_current = m_array.size() - 1;
}


//-----------------------------------------------------------------------------

// Set iterator to the next element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Iterator::to_next()
{
    ++m_current;
}


//-----------------------------------------------------------------------------

// Set iterator to the previous element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Iterator::to_previous()
{
    --m_current;
}


//-----------------------------------------------------------------------------

// Return true if iterator is exhausted, false otherwise.
template <typename T, typename Allocator>
inline
bool Array<T, Allocator>::Iterator::at_end() const
{
    return m_array.size() <= m_current;
}


//-----------------------------------------------------------------------------

// Return reference to the current element.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Reference
    Array<T, Allocator>::Iterator::operator*()
{
    ASSERT(M_CONT, m_current < m_array.size());
    return m_array[m_current];
}


//-----------------------------------------------------------------------------

// Apply member selection to the current element.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Pointer
    Array<T, Allocator>::Iterator::operator->()
{
    return &(operator*());
}


//=============================================================================

//-----------------------------------------------------------------------------

// Constructor.
template <typename T, typename Allocator>
inline
Array<T, Allocator>::Const_iterator::Const_iterator(
    const Array<T, Allocator>& array)	// array to iterate over
  : m_array(array), m_current(0)
{}


//-----------------------------------------------------------------------------

// Destructor.
template <typename T, typename Allocator>
inline
Array<T, Allocator>::Const_iterator::~Const_iterator()
{}


//-----------------------------------------------------------------------------

// Set iterator to the first element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Const_iterator::to_first()
{
    m_current = 0;
}


//-----------------------------------------------------------------------------

// Set iterator to the last element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Const_iterator::to_last()
{
    m_current = m_array.size() - 1;
}


//-----------------------------------------------------------------------------

// Set iterator to the next element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Const_iterator::to_next()
{
    ++m_current;
}


//-----------------------------------------------------------------------------

// Set iterator to the previous element of the array if it exists.
template <typename T, typename Allocator>
inline
void Array<T, Allocator>::Const_iterator::to_previous()
{
    --m_current;
}


//-----------------------------------------------------------------------------

// Return true if iterator is exhausted, false otherwise.
template <typename T, typename Allocator>
inline
bool Array<T, Allocator>::Const_iterator::at_end() const
{
    return m_array.size() <= m_current;
}


//-----------------------------------------------------------------------------

// Return reference to the current element.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Const_reference
    Array<T, Allocator>::Const_iterator::operator*() const
{
    ASSERT(M_CONT, m_current < m_array.size());
    return m_array[m_current];
}


//-----------------------------------------------------------------------------

// Apply member selection to the current element.
template <typename T, typename Allocator>
inline
typename Array<T, Allocator>::Const_pointer
    Array<T, Allocator>::Const_iterator::operator->() const
{
    return &(operator*());
}


//=============================================================================

//-----------------------------------------------------------------------------

// See Array::swap().
template<typename T, typename Allocator>
inline
void swap(
    Array<T, Allocator>& one,		// the one
    Array<T, Allocator>& other)		// the other
{
    one.swap(other);
}


//=============================================================================

//-----------------------------------------------------------------------------

// See note in base/lib/mem/i_mem_consumption.h about ADL and built-in types

inline bool has_dynamic_memory_consumption (const unsigned int&) { return false; }
inline size_t dynamic_memory_consumption (const unsigned int&) { return 0; }
inline bool has_dynamic_memory_consumption (const unsigned char&) { return false; }
inline size_t dynamic_memory_consumption (const unsigned char&) { return 0; }
inline bool has_dynamic_memory_consumption (const float&) { return false; }
inline size_t dynamic_memory_consumption (const float&) { return 0; }
inline bool has_dynamic_memory_consumption (const double&) { return false; }
inline size_t dynamic_memory_consumption (const double&) { return 0; }

// See base/lib/mem/i_mem_consumption.h

template<typename T, typename Allocator>
inline bool has_dynamic_memory_consumption (const Array<T, Allocator>& the_array)
{
    return true;
}

template<typename T, typename Allocator>
inline size_t dynamic_memory_consumption (const Array<T, Allocator>& the_array)
{
    // static size of the array elements
    size_t total = the_array.capacity() * sizeof(T);

    // additional dynamic size of the array elements
    if (the_array.size() > 0 && has_dynamic_memory_consumption (the_array[0])) {
        const size_t n = the_array.size();
        for (size_t i = 0; i < n; ++i)
            total += dynamic_memory_consumption (the_array[i]);
    }

    return total;
}

} // namespace CONT
} // namespace MI
