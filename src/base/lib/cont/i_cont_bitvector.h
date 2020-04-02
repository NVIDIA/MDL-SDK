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
 /// \brief Provides bitvector class, a sequence of bits. 
 ///
 ///        The bits are accessed by index, one index for each bit.
 ///        Initially, all bits are cleared to zero.

#ifndef BASE_LIB_CONT_I_CONT_BITVECTOR_H
#define BASE_LIB_CONT_I_CONT_BITVECTOR_H

#include <base/system/main/types.h>
#include <base/system/main/i_module_id.h>
#include <cstring>
#include <base/lib/log/i_log_assert.h>	// for ASSERT

namespace MI {
namespace CONT {

class Bitvector {
  public:
    // constructor, clear bits
    explicit Bitvector(
	size_t	n_bits);			// bits to be allocated

    // copy constructor
    Bitvector(
	const Bitvector &other);		// copy this

    // assignment operator
    Bitvector &operator=(
	const Bitvector &other);		// set to this

    // destructor
    ~Bitvector();

    // set new size, new bits cleared
    void resize(
	size_t	n_bits);			// new number of bits

    // return true if index has set the flag
    bool is_set(
	size_t	index) const;			// bit index

    // set flag for a given index
    void set(
	size_t	index,				// bit index
	bool	value);				// flag for bit

    // set flag for a given index
    void set(
	size_t	index);				// bit index

    // clear all flags to zero
    void clear();

    // return number of bits in bitvector
    size_t size() const;

    // return number of internally allocated bits in bitvector
    size_t capacity() const;

    // return binary data
    const Uint8* get_binary_data() const;
    size_t       get_binary_size() const;

    // set from binary data
    void set_binary_data(
	const size_t	nbits,			// bits
	const Uint8*	data);			// data

    // get the number of bits of the underlying base data type
    static size_t get_num_base_type_bits();

  private:

    Uint8 *	m_data;				// bits
    size_t	m_nbits;			// number of bits in use
    size_t	m_nb8;				// number of Uint8 allocated
};
 
}}


#include "cont_bitvector_inline.h"

#endif // #ifndef BASE_LIB_CONT_I_CONT_BITVECTOR_H
