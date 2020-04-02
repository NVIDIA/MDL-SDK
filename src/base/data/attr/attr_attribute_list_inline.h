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
/// \brief Attribute_list class inline implementation.
 
namespace MI {
namespace ATTR {


inline Uint Attribute_list::get_listsize() const
{
    return m_listsize;
}

inline Uint Attribute_list::get_listcapacity( ) const
{
    return m_listalloc;
}

inline const char *Attribute_list::get_list_values(
    Uint		i) const	// if attribute list, list index
{
    return get_values_i(i);
}

inline char *Attribute_list::set_list_values(
    Uint		i)		// if attribute list, list index
{
    return set_values_i(i);
}

inline bool Attribute_list::get_list_value_bool(
    Uint i,
    Uint n) const
{
    return get_value_bool_i(i, n);
}

inline void Attribute_list::set_list_value_bool(
    bool v,
    Uint i,
    Uint n)
{
    return set_value_bool_i(v, i, n);
}

inline int Attribute_list::get_list_value_int(
    Uint i,
    Uint n) const
{
    return get_value_int_i(i, n);
}

inline void Attribute_list::set_list_value_int(
    int v,
    Uint i,
    Uint n)
{
    return set_value_int_i(v, i, n);
}

inline Scalar Attribute_list::get_list_value_scalar(
    Uint i,
    Uint n) const
{
    return get_value_scalar_i(i, n);
}

inline Attribute_list::Vector3 Attribute_list::get_list_value_vector3(
    Uint i,
    Uint n) const
{
    return get_value_vector3_i(i, n);
}

inline mi::math::Color Attribute_list::get_list_value_color(
    Uint i,
    Uint n) const
{
    return get_value_color_i(i, n);
}

inline void Attribute_list::set_list_value_scalar(
    Scalar v,
    Uint i,
    Uint n)
{
    return set_value_scalar_i(v, i, n);
}

inline void Attribute_list::set_list_value_string(
    const char* v,
    Uint i,
    Uint n)
{
    return set_value_string_i(v, i, n);
}

inline void Attribute_list::set_list_value_vector3(
    const Vector3& v,
    Uint i,
    Uint n)
{
    return set_value_vector3_i(v, i, n);
}

inline const char* Attribute_list::get_list_value_string(
    Uint i,
    Uint n) const
{
    return get_value_string_i(i, n);
}

template <typename T>
inline const T& Attribute_list::get_list_value_ref(  Uint i, Uint n ) const
{
    return get_value_ref_i<T>(i, n);
}

template <typename T>
inline const T Attribute_list::get_list_value(  Uint i, Uint n ) const
{
    return get_list_value_ref<T>(i, n);
}

template <typename T>
inline void Attribute_list::set_list_value( const T &v, Uint i, Uint n )
{
    set_value_i(v, i, n);
}


// swaps this and another Attribute_list
// This function swaps two Attributes by exchanging the data in constant time.
inline void Attribute_list::swap(
    Attribute_list& other)			// swap with this
{
    Parent_type::swap(other);
    std::swap(m_listsize, other.m_listsize);
    std::swap(m_listalloc, other.m_listalloc);
}


// See Attribute_list::swap().
inline void swap(
    Attribute_list& one,		// The one
    Attribute_list& other)		// The other
{
    one.swap(other);
}


}
}
