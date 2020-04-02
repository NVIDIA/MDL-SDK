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
/// \brief inlined Attribute functions
 

#include <algorithm>


namespace MI {
namespace ATTR {

//
// access functions. There are no set members because of interdependencies.
// Note that the attribute name is actually stored in the Type.
//

inline Attribute_id Attribute::get_id() const
{
    return m_id;
}


inline const Type &Attribute::get_type() const
{
    return m_type;
}


inline Attribute_propagation Attribute::get_override() const
{
    return m_override;
}


inline void Attribute::set_override(
    Attribute_propagation	ov)		// override children?
{
    m_override = ov;
}


inline bool Attribute::get_global() const
{
    return m_global;
}


inline void Attribute::set_global(
    bool		global)			// participates in inheritance?
{
    m_global = global;
}


//
// return the list of attachments
//

inline const CONT::Array<Attachment> &Attribute::get_attachments() const
{
    return m_attachments;
}


//
// access values. get_values returns a pointer to the beginning of the data
// block, and the caller must figure out the layout from the type tree. The
// others are simple convenience functions that provide read-only access to
// values of common predefined types. Notice how I cleverly resist the
// temptation to use a template instead of casts by insisting on type asserts.
//

inline const char* Attribute::get_values() const
{
    return m_values;
}

inline char* Attribute::set_values()
{
    return m_values;
}

inline bool Attribute::get_value_bool(
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_BOOLEAN));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return n >= m_type.get_arraysize() ? 0 : reinterpret_cast<const bool*>(m_values)[n];
}


inline void Attribute::set_value_bool(
    bool		v,		// new value to set
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_BOOLEAN));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n < m_type.get_arraysize())
        reinterpret_cast<bool*>(m_values)[n] = v;
}


inline int Attribute::get_value_int(
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
                || contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support Scalar as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32 || m_type.get_typecode() == TYPE_ENUM)
        return reinterpret_cast<const int*>(m_values)[n];
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return int(reinterpret_cast<const Scalar*>(m_values)[n]);
    else
        return 0;
}

inline void Attribute::set_value_int(
    int			v,		// new value to set
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
        || contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return;

    // be kinda tolerant to support old mr syntax, ie support Scalar as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32 || m_type.get_typecode() == TYPE_ENUM)
        reinterpret_cast<int*>(m_values)[n] = v;
    else if (m_type.get_typecode() == TYPE_SCALAR)
        reinterpret_cast<float*>(m_values)[n] = float(v);
}

inline Scalar Attribute::get_value_scalar(
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
        || contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support int as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        return Scalar(reinterpret_cast<const int*>(m_values)[n]);
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return reinterpret_cast<const Scalar*>(m_values)[n];
    else
        return 0;
}

inline Dscalar Attribute::get_value_dscalar(
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
        || contains_expected_type(m_type, TYPE_SCALAR)
        || contains_expected_type(m_type, TYPE_DSCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support int as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        return Scalar(reinterpret_cast<const int*>(m_values)[n]);
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return reinterpret_cast<const Scalar*>(m_values)[n];
    else if (m_type.get_typecode() == TYPE_DSCALAR)
        return reinterpret_cast<const Dscalar*>(m_values)[n];
    else
        return 0;
}

inline Attribute::Vector3 Attribute::get_value_vector3(
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_VECTOR3));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return (m_type.get_typecode() == TYPE_VECTOR3 && n < m_type.get_arraysize()) ?
            reinterpret_cast<const Vector3*>(m_values)[n] :
            Vector3(0);
}

inline mi::math::Color Attribute::get_value_color(
    Uint		n) const
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_COLOR) ||
                   contains_expected_type(m_type, TYPE_VECTOR3) ||
                   contains_expected_type(m_type, TYPE_VECTOR4));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return mi::math::Color(0.f);
    // be kinda tolerant, ie support Vector3 / Vector4 as well
    if (m_type.get_typecode() == TYPE_COLOR)
        return reinterpret_cast<const mi::math::Color*>(m_values)[n];
    else if (m_type.get_typecode() == TYPE_VECTOR3) {
        const Vector3& v = reinterpret_cast<const Vector3*>(m_values)[n];
        return mi::math::Color(v.x, v.y, v.z);
    }
    else if (m_type.get_typecode() == TYPE_VECTOR4) {
        const Vector4& v = reinterpret_cast<const Vector4*>(m_values)[n];
        return mi::math::Color(v.x, v.y, v.z, v.w);
    }
    else
        return mi::math::Color(0.f);
}


inline void Attribute::set_value_scalar(
    Scalar		v,		// new value to set
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (n >= m_type.get_arraysize())
        return;

    // be kinda tolerant to support old mr syntax, ie support int as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        reinterpret_cast<int*>(m_values)[n] = static_cast<int>(v);
    else
        reinterpret_cast<float*>(m_values)[n] = v;
}


inline void Attribute::set_value_dscalar(
    Dscalar		v,		// new value to set
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_DSCALAR));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n < m_type.get_arraysize())
        reinterpret_cast<Dscalar*>(m_values)[n] = v;
}

inline void Attribute::set_value_vector3(
    const Vector3& v,
    Uint n)
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_VECTOR3));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n < m_type.get_arraysize())
        reinterpret_cast<Vector3*>(m_values)[n] = v;
}

inline const char* Attribute::get_value_string(
    Uint n) const
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_STRING));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (n >= m_type.get_arraysize())
         return 0;
    return reinterpret_cast<const char* const*>(m_values)[n];
}

template <typename T>
inline const T& Attribute::get_value_ref(Uint n) const
{
    ASSERT(M_ATTR, contains_expected_type(m_type, Type_code_traits<T>::type_code));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return reinterpret_cast<const T*>(m_values)[n];
}

template <typename T>
inline T Attribute::get_value(Uint n) const
{
    return get_value_ref<T>(n);
}

// specialization for bool
template <>
inline bool Attribute::get_value<bool>(Uint n) const
{
    return get_value_bool(n);
}

// specialization for int
template <>
inline int Attribute::get_value<int>(Uint n) const
{
    return get_value_int(n);
}

// specialization for float
template <>
inline float Attribute::get_value<float>(Uint n) const
{
    return get_value_scalar(n);
}

// specialization for double
template <>
inline double Attribute::get_value<double>(Uint n) const
{
    return get_value_dscalar(n);
}

// specialization for Vector3
template <>
inline Attribute::Vector3 Attribute::get_value<Attribute::Vector3>(Uint n) const
{
    return get_value_vector3(n);
}

// specialization for const char*
template <>
inline const char* Attribute::get_value<const char*>(Uint n) const
{
    return get_value_string(n);
}

// specialization for Color
template <>
inline mi::math::Color Attribute::get_value<mi::math::Color>(Uint n) const
{
    return get_value_color(n);
}

template <typename T>
inline void Attribute::set_value( const T &v, Uint n)
{
    ASSERT(M_ATTR, contains_expected_type(m_type, Type_code_traits<T>::type_code));
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    reinterpret_cast<T*>(m_values)[n] = v;
}


inline const char *Attribute::get_values_i(
    Uint		i) const	// if attribute list, list index
{
    ASSERT(M_ATTR, i==0 || i < get_listsize());
    return i >= get_listsize() ? 0 :
           i ? m_values + i * m_type.sizeof_all() : m_values;
}

inline char *Attribute::set_values_i(
    Uint		i)		// if attribute list, list index
{
    ASSERT(M_ATTR, i < get_listsize());
    return i >= get_listsize() ? 0 :
           i ? m_values + i * m_type.sizeof_all() : m_values;
}

inline bool Attribute::get_value_bool_i(
    Uint		i,		// if attribute list, list index
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_BOOLEAN));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return i >= get_listsize() || n >= m_type.get_arraysize() ? 0 :
           reinterpret_cast<const bool*>(m_values)[i * m_type.get_arraysize() + n];
}


inline void Attribute::set_value_bool_i(
    bool		v,		// new value to set
    Uint		i,		// if attribute list, list index
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_BOOLEAN));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i < get_listsize() && n < m_type.get_arraysize())
        reinterpret_cast<bool*>(m_values)[i * m_type.get_arraysize() + n] = v;
}


inline int Attribute::get_value_int_i(
    Uint		i,		// if attribute list, list index
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
                || contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (i >= get_listsize() || n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support Scalar as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        return reinterpret_cast<const int*>(m_values)[i * m_type.get_arraysize() + n];
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return int(reinterpret_cast<const Scalar*>(m_values)[i * m_type.get_arraysize() + n]);
    else
        return 0;
}


inline void Attribute::set_value_int_i(
    int			v,		// new value to set
    Uint		i,		// if attribute list, list index
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i < get_listsize() && n < m_type.get_arraysize())
        reinterpret_cast<int*>(m_values)[i * m_type.get_arraysize() + n] = v;
}

inline Scalar Attribute::get_value_scalar_i(
    Uint		i,		// if attribute list, list index
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
        || contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (i >= get_listsize() || n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support int as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        return Scalar(reinterpret_cast<const int*>(m_values)[i * m_type.get_arraysize() + n]);
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return reinterpret_cast<const Scalar*>(m_values)[i * m_type.get_arraysize() + n];
    else
        return 0;
}

inline Dscalar Attribute::get_value_dscalar_i(
    Uint		i,		// if attribute list, list index
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_INT32)
        || contains_expected_type(m_type, TYPE_SCALAR)
        || contains_expected_type(m_type, TYPE_DSCALAR));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (i >= get_listsize() || n >= m_type.get_arraysize())
        return 0;
    // be kinda tolerant, ie support int as well by simple cast
    if (m_type.get_typecode() == TYPE_INT32)
        return Scalar(reinterpret_cast<const int*>(m_values)[i * m_type.get_arraysize() + n]);
    else if (m_type.get_typecode() == TYPE_SCALAR)
        return reinterpret_cast<Scalar*>(m_values)[i * m_type.get_arraysize() + n];
    else if (m_type.get_typecode() == TYPE_DSCALAR)
        return reinterpret_cast<Dscalar*>(m_values)[i * m_type.get_arraysize() + n];
    else
        return 0;
}

inline Attribute::Vector3 Attribute::get_value_vector3_i(
    Uint		i,		// if attribute list, list index
    Uint		n) const	// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_VECTOR3));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return (m_type.get_typecode() == TYPE_VECTOR3 &&
            i < get_listsize() &&
            n < m_type.get_arraysize()) ?
            reinterpret_cast<const Vector3*>(m_values)[i * m_type.get_arraysize() + n] :
            Vector3(0);
}

inline mi::math::Color Attribute::get_value_color_i(
    Uint		i,
    Uint		n) const
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_COLOR) ||
                   contains_expected_type(m_type, TYPE_VECTOR3) ||
                   contains_expected_type(m_type, TYPE_VECTOR4));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());

    if (i >= get_listsize() || n >= m_type.get_arraysize())
        return mi::math::Color(0.f);
    // be kinda tolerant, ie support Vector3 / Vector4 as well
    if (m_type.get_typecode() == TYPE_COLOR)
        return reinterpret_cast<const mi::math::Color*>(m_values)[i * m_type.get_arraysize() + n];
    else if (m_type.get_typecode() == TYPE_VECTOR3)
    {
        const Vector3& v = 
                reinterpret_cast<const Vector3*>(m_values)[i * m_type.get_arraysize() + n];
        return mi::math::Color(v.x, v.y, v.z);
    }
    else if (m_type.get_typecode() == TYPE_VECTOR4)
    {
        const Vector4& v = 
                reinterpret_cast<const Vector4*>(m_values)[i * m_type.get_arraysize() + n];
        return mi::math::Color(v.x, v.y, v.z, v.w);
    }
    else
        return mi::math::Color(0.f);
}

inline void Attribute::set_value_scalar_i(
    Scalar		v,		// new value to set
    Uint		i,		// if attribute list, list index
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_SCALAR));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i < get_listsize() && n < m_type.get_arraysize())
        reinterpret_cast<Scalar*>(m_values)[i * m_type.get_arraysize() + n] = v;
}

inline void Attribute::set_value_dscalar_i(
    Dscalar		v,		// new value to set
    Uint		i,		// if attribute list, list index
    Uint		n)		// if array, get n-th value
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_DSCALAR));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i < get_listsize() && n < m_type.get_arraysize())
        reinterpret_cast<Dscalar*>(m_values)[i * m_type.get_arraysize() + n] = v;
}

inline void Attribute::set_value_vector3_i(
    const Vector3&	v,
    Uint		i,
    Uint		n)
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_VECTOR3));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i < get_listsize() && n < m_type.get_arraysize())
        reinterpret_cast<Vector3*>(m_values)[i * m_type.get_arraysize() + n] = v;
}

inline const char* Attribute::get_value_string_i(
    Uint            i,
    Uint            n)
    const
{
    ASSERT(M_ATTR, contains_expected_type(m_type, TYPE_STRING));
    ASSERT(M_ATTR, i < get_listsize() );
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    if (i >= get_listsize() || n >= m_type.get_arraysize())
         return 0;
    return reinterpret_cast<const char* const*>(m_values)[ i * m_type.get_arraysize() + n];
}

template <typename T>
inline const T& Attribute::get_value_ref_i(  Uint i, Uint n ) const
{
    ASSERT(M_ATTR, contains_expected_type( m_type, Type_code_traits<T>::type_code));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    return reinterpret_cast<const T*>(m_values)[ i * m_type.get_arraysize() + n ];
}

template <typename T>
inline T Attribute::get_value_i(  Uint i, Uint n ) const
{
    return get_value_ref_i<T>(i, n);
}

// specialization for bool
template <>
inline bool Attribute::get_value_i<bool>( Uint i, Uint n ) const
{
    return get_value_bool_i(i, n);
}

// specialization for int
template <>
inline int Attribute::get_value_i<int>( Uint i, Uint n ) const
{
    return get_value_int_i(i, n);
}

// specialization for float
template <>
inline float Attribute::get_value_i<float>( Uint i, Uint n ) const
{
    return get_value_scalar_i(i, n);
}

// specialization for double
template <>
inline double Attribute::get_value_i<double>( Uint i, Uint n ) const
{
    return get_value_dscalar_i(i, n);
}

// specialization for Vector3
template <>
inline Attribute::Vector3 Attribute::get_value_i<Attribute::Vector3>( Uint i, Uint n ) const
{
    return get_value_vector3_i(i, n);
}

// specialization for const char*
template <>
inline const char* Attribute::get_value_i<const char*>( Uint i, Uint n ) const
{
    return get_value_string_i(i, n);
}

// specialization for Color
template <>
inline mi::math::Color Attribute::get_value_i<mi::math::Color>( Uint i, Uint n ) const
{
    return get_value_color_i(i, n);
}

template <typename T>
inline void Attribute::set_value_i( const T &v, Uint i, Uint n )
{
    ASSERT(M_ATTR, contains_expected_type(m_type, Type_code_traits<T>::type_code));
    ASSERT(M_ATTR, i < get_listsize());
    ASSERT(M_ATTR, n < m_type.get_arraysize());
    reinterpret_cast<T*>(m_values)[ i * m_type.get_arraysize() + n ] = v;
}


//
// make a copy of this attribute, return a pointer to it
//

inline Attribute *Attribute::copy() const
{
    Attribute *attr = new Attribute(*this);
    return attr;
}


//
// swap this Attribute with another Attribute.
// This function swaps two attributes by exchanging the values data block, id,
// type and flags, which is done in constant time. Note that the global swap()
// function falls back to this function due to its template specialization.
//

inline void Attribute::swap(
    Attribute		&other)		// the other attribute
{
    using std::swap;
    swap(m_id,		other.m_id);
    swap(m_override,	other.m_override);
    // should the Type swap be a fast swap, as well ? so far: just copy it
    swap(m_type,	other.m_type);
    swap(m_values,	other.m_values);
    swap(m_attachments,	other.m_attachments);
    swap(m_global,	other.m_global);
}


//
// unique class ID so that the receiving host knows which class to create
//

inline SERIAL::Class_id Attribute::get_class_id() const
{
    return id;
}

//
// check, if this attribute is of the given type. This is true, if either
// the class id of this attribute equals the given class id, or the class
// is derived from another class which has the given class id.
//

inline bool Attribute::is_type_of(
    SERIAL::Class_id id) const		// the class id to check
{
    return id == ID_ATTRIBUTE ? true : false;
}


//
// see Attribute::swap().
//

inline void swap(
    Attribute	&one,			// the one
    Attribute	&other)			// the other
{
    one.swap(other);
}


} // namespace ATTR
} // namespace MI
