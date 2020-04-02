/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation of the ATTR::Type_iterator inline members

namespace MI {
namespace ATTR {

//
// let the iterator pointer to the given type and value
//

inline void Type_iterator::set(
    const Type 		*type,		// current type
    char 		*values)	// pointer to data of the type instance
{
    m_type	      = type;
    m_value	      = values;
    if (m_type && m_type->get_typecode() == TYPE_UNDEF)
        m_type = 0;
}


//
// iterate over the linear list of elements of a Leaf_func. If a returned
// element is a struct or array (or both), the caller must recurse using
// the second constructor. Only the first member of an array is considered.
//
// constructor: iterate over all toplevel elements of a Leaf_func. Start
// by moving to the first element at offset 0. If that typer is UNDEF, the
// shader has no elements, clear m_type so at_end() is true immediately.
// Attaching subshaders is not yet supported.
//

inline Type_iterator::Type_iterator(
    const Type		*type,		// current type
    char		*values)	// data
{
    set(type, values);
}


//
// constructor: iterate over the sub-elements of a struct element.
// If the struct is not an array or a fixed array, its first member has the
// same address as the struct. If it's a dynamic array, the Dynamic_array
// record points to the values of the dynamic array. Note that only the first
// member of an array is visited during the iteration.
// NOTE The current implementation for TYPE_CALL iterators goes straight to the child. Hence
// the passed in values MUST point to the struct's values, ie plus 2*Type::sizeof_one(TYPE_STRING).
//

inline Type_iterator::Type_iterator(
    Type_iterator	*par,		// struct to recurse into
    char		*values)	// data
{
    ASSERT(M_ATTR, par->m_type->get_typecode() == TYPE_STRUCT
                || par->m_type->get_typecode() == TYPE_ARRAY
                || par->m_type->get_typecode() == TYPE_ATTACHABLE
                || par->m_type->get_typecode() == TYPE_CALL);
    m_type		  = par->m_type->get_child();
    m_value 		  = values;
}


//
// skip forward to the next element. First add the size of the current
// element to the value pointer, then align properly for the next element.
// For example, stepping from bool to int means adding 1 and then aligning to
// the next 4-byte address boundary.
//

inline void Type_iterator::to_next()
{
    m_value += m_type->sizeof_all();
    m_type = m_type->get_next();
    if (m_type) {
        size_t align = m_type->align_all()-1;
        m_value   = (char *)((size_t(m_value) + align) & ~align);
    }
}

//
// at end of element chain?
//

inline bool Type_iterator::at_end() const
{
    return !m_type;
}


//
// access functions for the current element. The caller needs to cast the
// return value from get_value to the type indicated by get_typecode because
// the type can be different for each step of the iteration. If get_arraysize
// ==0, get_value returns a pointer to Dynamic_array.
//

inline const char *Type_iterator::get_name() const
{
    return m_type->get_name();
}


inline ATTR::Type_code Type_iterator::get_typecode(
    bool array_elem_type) const
{
    if (!array_elem_type)
        return m_type->get_typecode();
    else
        return m_type->get_typecode() == TYPE_ARRAY && m_type->get_child()?
            m_type->get_child()->get_typecode() :
            TYPE_UNDEF;
}


inline const Type* Type_iterator::operator->() const
{
    return m_type;
}


inline char *Type_iterator::get_value() const
{
    return m_value;
}


inline int Type_iterator::get_arraysize() const
{
    return m_type->get_arraysize();
}

inline size_t Type_iterator::sizeof_elem() const
{
    return m_type->sizeof_elem();
}

}
}
