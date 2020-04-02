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
 /// \brief IImplementation of the rle compression array.
 ///
 ///        Rle_arrays implement sequences of dynamically varying length, which
 ///        are compressed by a simple RLE sceme. This class is parameterized by
 ///        the type of the values and possibly by the underlying techniques.

namespace MI
{
namespace CONT
{


//=============================================================================

// Default constructor.
template <typename T, typename CONT, typename IND>
inline
Rle_array<T, CONT, IND>::Rle_array()
  : m_data(0), m_index(0)
{}


//-----------------------------------------------------------------------------

// Add n times item into array.
template <typename T, typename CONT, typename IND>
inline
void Rle_array<T, CONT, IND>::push_back(
    const T& item,			// datum to be inserted
    size_t n)				// number of insertions
{
    if (n == 0)
	return;

    // handle special case, erm, special
    if (empty()) {
        //assert(m_data.empty());
        m_data.append(item);
        m_index.append(n);

        return;
    }

    // compare last element with item
    if (m_data[m_index.size() - 1] == item)
        // increase index only
        m_index[m_index.size()-1] += n;
    else {
        // insert both new item and index
        m_index.append(m_index[m_index.size()-1]+n);
        m_data.append(item);
    }
}


//-----------------------------------------------------------------------------

// Remove last item from array.
template <typename T, typename CONT, typename IND>
inline
void Rle_array<T, CONT, IND>::pop_back()
{
    ASSERT(M_CONT, !empty());

    // compute number of items of last element by the difference of indices
    size_t count = 0;
    if (m_index.size() == 1)
        count = m_index[0];
    else
        count = m_index[m_index.size() - 1] - m_index[m_index.size() - 2];

    // remove according to the number of items
    if (count == 1) {
//         m_index.pop_back();
//         m_data.pop_back();
	m_index.remove(m_index.size()-1);
	m_data.remove(m_data.size()-1);
    }
    else
        --m_index[m_index.size() - 1];
}


//-----------------------------------------------------------------------------

// Return the i-th item in the array.
template <typename T, typename CONT, typename IND>
inline
const T& Rle_array<T, CONT, IND>::operator[](
    size_t id) const			// index
{
    ASSERT(M_CONT, !empty());

    // binary search of the right index:
    // we're looking for the first index bigger than id
    int l = 0;
    int h = (int)m_index.size() - 1;
    int m = 0;
    while (l <= h)
    {
	m = (l + h) / 2;
	if (m_index[m] == id)
	{
	    ASSERT(M_CONT, m < (int)m_index.size()-1);
	    return m_data[m+1];
	}
	if (m_index[m] > id)
	    h = m-1;
	else
	    l = m+1;
    }
    // If we reached this point, l > h.
    // l should now be the right index.
    if (l > (int)m_index.size()-1) // Not sure, but this might never happen...
	l = (int)m_index.size()-1;
    ASSERT(M_CONT, m_index[l] > id);
    return m_data[l];
}


//-----------------------------------------------------------------------------

// Return the first item in the array.
template <typename T, typename CONT, typename IND>
inline
const T& Rle_array<T, CONT, IND>::front() const
{
    ASSERT(M_CONT, !empty());
    return m_data[0];
}


//-----------------------------------------------------------------------------

// Return the last item in the array.
template <typename T, typename CONT, typename IND>
inline
const T& Rle_array<T, CONT, IND>::back() const
{
    ASSERT(M_CONT, !empty());
    return m_data[m_data.size()-1];
}


//-----------------------------------------------------------------------------

// Is array empty?
template <typename T, typename CONT, typename IND>
inline
bool Rle_array<T, CONT, IND>::empty() const
{
    return m_index.empty();
}


//-----------------------------------------------------------------------------

// Get number of items in array.
template <typename T, typename CONT, typename IND>
inline
size_t Rle_array<T, CONT, IND>::size() const
{
    //return empty()? 0 : m_index.back();
    return empty()? 0 : m_index[m_index.size()-1];
}

//-----------------------------------------------------------------------------

// Erase all elements in array.
template <typename T, typename CONT, typename IND>
inline
void Rle_array<T, CONT, IND>::clear()
{
    m_data.clear();
    m_index.clear();
}

//-----------------------------------------------------------------------------

// Retrieve the size of the index array. This implementation detail is used
// by the iterator and internally.
template <typename T, typename CONT, typename IND>
inline
size_t Rle_array<T, CONT, IND>::get_index_size() const
{
    return m_index.size();
}

//-----------------------------------------------------------------------------

// Retrieve the byte size of the array.
template <typename T, typename CONT, typename IND>
inline
size_t Rle_array<T, CONT, IND>::get_byte_size() const
{
    return m_index.size() * (sizeof(typename IND::value_type) + sizeof(T));
}

//-----------------------------------------------------------------------------

// Return iterator to the first item.
template <typename T, typename CONT, typename IND>
inline
typename Rle_array<T, CONT, IND>::Const_iterator
Rle_array<T, CONT, IND>::begin() const
{
    // this definition of first iterator was choosen by me
    return Const_iterator(this, 0, 0);
}


//-----------------------------------------------------------------------------

// Return iterator behind the very end of the array.
template <typename T, typename CONT, typename IND>
inline
typename Rle_array<T, CONT, IND>::Const_iterator
Rle_array<T, CONT, IND>::end() const
{
    // this definition of one behind the last iterator was choosen by me
    return Const_iterator(this, m_index.size(), 0);
}


//-----------------------------------------------------------------------------

// Return iterator to the first chunk.
template <typename T, typename CONT, typename IND>
inline
typename Rle_array<T, CONT, IND>::Chunk_iterator
Rle_array<T, CONT, IND>::begin_chunk() const
{
    return Chunk_iterator(*this, 0);
}


//-----------------------------------------------------------------------------

// Return iterator behind the very end of the array.
template <typename T, typename CONT, typename IND>
inline
typename Rle_array<T, CONT, IND>::Chunk_iterator
Rle_array<T, CONT, IND>::end_chunk() const
{
    return Chunk_iterator(*this, m_index.size());
}


//=============================================================================

//-----------------------------------------------------------------------------

// Constructor.
template <typename T, typename CONT, typename IND>
inline
Rle_iterator<T, CONT, IND>::Rle_iterator(
    const Rle_array<T, CONT, IND>* array, // the array to work on
    size_t index,			// start index into index array
    size_t data)			// start index into data array
  : m_array(array), m_index(index), m_offset(data)
{}


//-----------------------------------------------------------------------------

// Copy Constructor.
template <typename T, typename CONT, typename IND>
inline
Rle_iterator<T, CONT, IND>::Rle_iterator(
    const Rle_iterator& other)		// the other
  : m_array(other.m_array), m_index(other.m_index), m_offset(other.m_offset)
{}


//-----------------------------------------------------------------------------

// Assignment operator.
template <typename T, typename CONT, typename IND>
inline
Rle_iterator<T, CONT, IND>& Rle_iterator<T, CONT, IND>::operator=(
    const Rle_iterator& other)		// the other
{
    if (&other != this) {
	m_array = other.m_array;
	m_index = other.m_index;
	m_offset= other.m_offset;
    }
    return *this;
}


//-----------------------------------------------------------------------------

// Preincrement.
template <typename T, typename CONT, typename IND>
inline
Rle_iterator<T, CONT, IND>& Rle_iterator<T, CONT, IND>::operator++()
{
    // choice between one check in every version or an ASSERT only?
    // speed vs safety - safety one for now.
    // TO DO: Parameterize this decision as a policy - SAFE vs UNSAFE
    if (!m_array)
	return *this;

    if (++m_offset >= m_array->m_index[m_index])
	++m_index;
    return *this;
}


//-----------------------------------------------------------------------------

// Postincrement.
template <typename T, typename CONT, typename IND>
inline
const Rle_iterator<T, CONT, IND> Rle_iterator<T, CONT, IND>::operator++(
    int)				// dummy
{
    Iterator tmp = *this;
    ++*this;
    return tmp;
}


//-----------------------------------------------------------------------------

// Predecrement.
template <typename T, typename CONT, typename IND>
inline
Rle_iterator<T, CONT, IND>& Rle_iterator<T, CONT, IND>::operator--()
{
    // TO DO: Parameterize this decision as a policy - SAFE vs UNSAFE
    if (!m_array)
	return *this;

    if (--m_offset <= m_array->m_index[m_index])
	--m_index;
    return *this;
}


//-----------------------------------------------------------------------------

// Postdecrement.
template <typename T, typename CONT, typename IND>
inline
const Rle_iterator<T, CONT, IND> Rle_iterator<T, CONT, IND>::operator--(
    int)				// dummy
{
    Iterator tmp = *this;
    --*this;
    return tmp;
}


//-----------------------------------------------------------------------------

// Return designated object.
template <typename T, typename CONT, typename IND>
inline
const T& Rle_iterator<T, CONT, IND>::operator*() const
{
    ASSERT(M_CONT, m_array && !m_array->empty());

    return m_array->m_data[m_index];
}


//-----------------------------------------------------------------------------

// Return pointer to class object.
template <typename T, typename CONT, typename IND>
inline
const T* Rle_iterator<T, CONT, IND>::operator->() const
{
    return *this;
}


//-----------------------------------------------------------------------------

// Comparison.
template <typename T, typename CONT, typename IND>
inline
bool Rle_iterator<T, CONT, IND>::operator==(
    const Rle_iterator<T, CONT, IND>& other) const // the other one
{
    // TO DO: Parameterize this decision as a policy - SAFE vs UNSAFE
    if (!m_array || !other.m_array)
	return  m_array == other.m_array;

    // special handling for end
    if (m_index == m_array->get_index_size() ||
        other.m_index == other.m_array->get_index_size())
        return m_index == other.m_index;

    return m_index == other.m_index && m_offset == other.m_index;
}


//-----------------------------------------------------------------------------

// Unequality comparison.
template <typename T, typename CONT, typename IND>
inline
bool Rle_iterator<T, CONT, IND>::operator!=(
    const Rle_iterator<T, CONT, IND>& other) const // the other one
{
    return (!(*this == other));
}


//=============================================================================

//-----------------------------------------------------------------------------

// Constructor.
template <typename T, typename CONT, typename IND>
inline
Rle_chunk_iterator<T, CONT, IND>::Rle_chunk_iterator(
    const Rle_array<T, CONT, IND>& array, // the array to work on
    size_t index)				// start index into index array
  : m_array(array), m_index(index)
{}


//-----------------------------------------------------------------------------

// Preincrement.
template <typename T, typename CONT, typename IND>
inline
Rle_chunk_iterator<T, CONT, IND>&
Rle_chunk_iterator<T, CONT, IND>::operator++()
{
    ++m_index;
    return *this;
}


//-----------------------------------------------------------------------------

// Postincrement.
template <typename T, typename CONT, typename IND>
inline
const Rle_chunk_iterator<T, CONT, IND>
Rle_chunk_iterator<T, CONT, IND>::operator++(
    int)				// dummy
{
    Iterator tmp = *this;
    ++*this;
    return tmp;
}


//-----------------------------------------------------------------------------

// Predecrement.
template <typename T, typename CONT, typename IND>
inline
Rle_chunk_iterator<T, CONT, IND>&
Rle_chunk_iterator<T, CONT, IND>::operator--()
{
    --m_index;
    return *this;
}


//-----------------------------------------------------------------------------

// Postdecrement.
template <typename T, typename CONT, typename IND>
inline
const Rle_chunk_iterator<T, CONT, IND>
Rle_chunk_iterator<T, CONT, IND>::operator--(
    int)				// dummy
{
    Iterator tmp = *this;
    --*this;
    return tmp;
}


//-----------------------------------------------------------------------------

// Return designated object.
template <typename T, typename CONT, typename IND>
inline
const Rle_chunk_iterator<T, CONT, IND>&
Rle_chunk_iterator<T, CONT, IND>::operator*() const
{
    ASSERT(M_CONT, !m_array.empty());
    return *this;//m_array.m_data[m_index];
}


//-----------------------------------------------------------------------------

// Return pointer to iterator object.
template <typename T, typename CONT, typename IND>
inline
const Rle_chunk_iterator<T, CONT, IND>*
Rle_chunk_iterator<T, CONT, IND>::operator->() const
{
    return &(operator*());
}


//-----------------------------------------------------------------------------

// Return the number of (consecutive) occurrences of the current item.
template <typename T, typename CONT, typename IND>
inline
size_t Rle_chunk_iterator<T, CONT, IND>::count() const
{
    if (m_array.empty())
	return 0;
    if (m_index == 0)
	return m_array.m_index[0];
    else
	return m_array.m_index[m_index] -  m_array.m_index[m_index-1];
}


//-----------------------------------------------------------------------------

// Return the current item.
template <typename T, typename CONT, typename IND>
inline
const T& Rle_chunk_iterator<T, CONT, IND>::data() const
{
    return m_array.m_data[m_index];
}



//-----------------------------------------------------------------------------

// Comparison.
template <typename T, typename CONT, typename IND>
inline
bool Rle_chunk_iterator<T, CONT, IND>::operator==(
    const Rle_chunk_iterator<T, CONT, IND>& other) const // the other one
{
    return &m_array == &other.m_array && m_index == other.m_index;
}


//-----------------------------------------------------------------------------

// Unequality comparison.
template <typename T, typename CONT, typename IND>
inline
bool Rle_chunk_iterator<T, CONT, IND>::operator!=(
    const Rle_chunk_iterator<T, CONT, IND>& other) const // the other one
{
    return (!(*this == other));
}



}
}
