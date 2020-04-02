/******************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief  Provides bitvector class, a sequence of bits.

#include <base/lib/mem/i_mem_allocatable.h>

namespace
{
// return size of a Uint8 bitmap vector with n bits
inline size_t b8size(
    size_t	n)			// number of bits
{
    return (n + 7) / 8;
}

}

namespace MI {

namespace CONT {

// clear all flags to zero
inline void Bitvector::clear()
{
    memset(m_data, 0, sizeof(Uint8) * b8size(m_nbits));
}

// constructor
inline Bitvector::Bitvector(
    size_t	n_bits)				// bits to be allocated
  : m_nbits(n_bits), m_nb8(b8size(n_bits))
{
    m_data = MI::MEM::new_array<Uint8>(m_nb8);
    clear();
}

// copy constructor
inline Bitvector::Bitvector(
    const Bitvector &other)			// copy this
  : m_nbits(other.m_nbits), m_nb8(b8size(other.m_nbits))
{
    m_data = MI::MEM::new_array<Uint8>(m_nb8);
    memcpy(m_data, other.m_data, sizeof(Uint8) * m_nb8);
}

// Return number of bits in bitvector.
inline size_t Bitvector::size() const
{
    return size_t(m_nbits);
}

// Return internally allocated space in number of bits in bitvector.
inline size_t Bitvector::capacity() const
{
    return size_t(m_nb8 * 8);
}

// assignement operator
inline Bitvector &Bitvector::operator=(
    const Bitvector &other)			// assign this to other
{
    // check for self assignment
    if (this == &other)
	return *this;

    clear();
    // allocate more memory
    if (capacity() < other.size()) {
	MI::MEM::delete_array<Uint8>(m_data);
	m_nb8 = b8size(other.size());
	m_data = MI::MEM::new_array<Uint8>(m_nb8);
    }
    m_nbits = other.size();
    memcpy(m_data, other.m_data, sizeof(Uint8) * b8size(m_nbits));
    return *this;
}

// destructor
inline Bitvector::~Bitvector()
{
    MI::MEM::delete_array<Uint8>(m_data);
}

// set new size, new bits cleared
inline void Bitvector::resize(
    size_t	n_bits)				// new number of bits
{
    // allocate new data if necessary
    if (n_bits > capacity()) {
	Uint8 * newdata = MI::MEM::new_array<Uint8>(b8size(n_bits));
	memcpy(newdata, m_data, sizeof(Uint8) * b8size(m_nbits));
	MI::MEM::delete_array<Uint8>(m_data);
	m_nb8 = b8size(n_bits);
	m_data = newdata;
    }

    // clear the new bits, including the formerly unused bits of the last b8
    if (n_bits > m_nbits) {
	memset(m_data+b8size(m_nbits), 0,
	    sizeof(Uint8)*(m_nb8-b8size(m_nbits)));

	size_t b8index = m_nbits/8;
	for (size_t i = m_nbits - 8*b8index; i < 8; ++i)
	    m_data[b8index]  &= ~(1 << (i & 7));
    }
    m_nbits = n_bits;
}

// return true if index has set the flag
inline bool Bitvector::is_set(
    size_t	index) const			// bit index
{
    ASSERT(M_CONT, index < m_nbits);
    return (m_data[index/8] & (1 << (index & 7))) != 0;
}

// set flag for a given index
inline void Bitvector::set(
    size_t	index,				// bit index
    bool	value)				// flag for bit
{
    ASSERT(M_CONT, index < m_nbits);
    if (value)
	m_data[index/8] |= 1 << (index & 7);
    else
	m_data[index/8] &= ~(1 << (index & 7));
}

// set flag for a given index
inline void Bitvector::set(
    size_t	index)				// bit index
{
    ASSERT(M_CONT, index < m_nbits);
    m_data[index/8] |= 1 << (index & 7);
}

// return binary data
inline const Uint8* Bitvector::get_binary_data() const
{
    return m_data;
}
// return binary data
inline size_t   Bitvector::get_binary_size() const
{
    return m_nb8;
}

// set from binary data
inline void Bitvector::set_binary_data(
    const size_t	nbits,			// bits
    const Uint8*	data)			// data
{
    m_nb8 = b8size( nbits );
    m_data = MI::MEM::new_array<Uint8>(m_nb8);
    m_nbits = nbits;
    memcpy(m_data, data, sizeof(Uint8) * m_nb8);
}

// get the number of bits of the underlying base data type
inline size_t Bitvector::get_num_base_type_bits()
{
    return sizeof(Uint8) * 8;
}
 
}}	// namespace MI::CONT



