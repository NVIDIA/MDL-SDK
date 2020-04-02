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
 /// \brief Implements a rle compression array.
 ///
 ///        Rle_arrays implement sequences of dynamically varying length, which
 ///        are compressed by a simple RLE scheme. This class is parameterized by
 ///        the type of the values and possibly by the underlying techniques.

#ifndef BASE_LIB_CONT_RLE_ARRAY_H
#define BASE_LIB_CONT_RLE_ARRAY_H

#include "i_cont_array.h"

namespace MI {
namespace CONT {

//=============================================================================

// forward declaration
template <typename T, typename CONT, typename IND>
class Rle_array;


//=============================================================================

// The iterator definition for the following RLE array.
// This iterator has (deliberately) the same defaults as the array and
// supports const access only(!).
template <typename T, typename CONT=Array<T>, typename IND=Array<size_t> >
class Rle_iterator
{
  public:
    // Convenient typedef.
    typedef Rle_array<T, CONT, IND> The_array;
    typedef Rle_iterator<T, CONT, IND> Iterator;

    // Constructor.
    explicit Rle_iterator(
	const The_array* array=0,	// array to iterate on
	size_t index=0,			// start index in index array
	size_t data=0);			// start index in data array
    // Copy Constructor.
    Rle_iterator(
	const Rle_iterator& other);	// the other
    // Assignment operator.
    Rle_iterator& operator=(
	const Rle_iterator& other);	// the other

    // Preincrement.
    Iterator& operator++();
    // Postincrement.
    const Iterator operator++(int);
    // Predecrement.
    Iterator& operator--();
    // Postdecrement.
    const Iterator operator--(int);

    // Return designated object.
    const T& operator*() const;
    // Return pointer to class object.
    const T* operator->() const;

    // Comparison.
    bool operator==(
	const Iterator& other) const;	// the other one
    // Unequality comparison.
    bool operator!=(
	const Iterator& other) const;	// the other one

  private:
    const The_array* m_array;		// array to operate on
    size_t m_index;                     // index into index array
    size_t m_offset;                    // offset in m_index' data
};


//=============================================================================

// The compressed iterator definition for the following RLE array.
// This iterator has (deliberately) the same defaults as the array and
// supports the compressed iteration and const access only(!).
template <typename T, typename CONT=Array<T>, typename IND=Array<size_t> >
class Rle_chunk_iterator
{
  public:
    // Convenient typedefs.
    typedef Rle_array<T, CONT, IND> The_array;
    typedef Rle_chunk_iterator<T, CONT, IND> Iterator;

    // Constructor.
    Rle_chunk_iterator(
	const The_array& array,		// array to iterate on
	size_t index);			// start index in index array

    // Preincrement.
    Iterator& operator++();
    // Postincrement.
    const Iterator operator++(int);
    // Predecrement.
    Iterator& operator--();
    // Postdecrement.
    const Iterator operator--(int);

    // Return reference to object.
    const Iterator& operator*() const;
    // Return pointer to object.
    const Iterator* operator->() const;

    // Return the number of (consecutive) occurrences of the current item.
    size_t count() const;
    // Return the current item.
    const T& data() const;

    // Comparison.
    bool operator==(
	const Rle_chunk_iterator& other) const;	// the other one
    // Unequality comparison.
    bool operator!=(
	const Rle_chunk_iterator& other) const;	// the other one

  private:
    const The_array& m_array;		// array to operate on
    size_t m_index;                        // index into index array
};


//=============================================================================

// A RLE compression implementing array.
template <typename T, typename CONT=Array<T>, typename IND=Array<size_t> >
class Rle_array
{
    friend class Rle_iterator<T, CONT, IND>;
    friend class Rle_chunk_iterator<T, CONT, IND>;
  public:
    // Convenient typedef.
    typedef typename Rle_iterator<T, CONT, IND>::Iterator Const_iterator;
    typedef typename Rle_chunk_iterator<T, CONT, IND>::Iterator Chunk_iterator;

    // Default constructor.
    Rle_array();

    // Add another item into array.
    void push_back(
	const T& item,			// item to add
	size_t n=1);			// number of times to add
    // Remove last item from array.
    void pop_back();

    // Return the i-th item in the array.
    const T& operator[](
	size_t i) const;		// index of requested element
    // Return the first item in the array.
    const T& front() const;
    // Return the last item in the array.
    const T& back() const;

    // Is array empty?
    bool empty() const;
    // Retrieve number of items in array.
    size_t size() const;
    // Erase all elements of array.
    void clear();
    // Retrieve size of the index array.
    size_t get_index_size() const;
    // Retrieve size of the array
    size_t get_byte_size() const;

    // Return iterator to the first item.
    Const_iterator begin() const;
    // Return iterator behind the very end of the array.
    Const_iterator end() const;

    // Return iterator to the first chunk.
    Chunk_iterator begin_chunk() const;
    // Return iterator behind the very end of the array.
    Chunk_iterator end_chunk() const;

  private:
    CONT m_data;                        // data container
    IND m_index;                        // index container

};

}
}

#include "cont_rle_array_inline.h"

#endif
