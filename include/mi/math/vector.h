/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/math/vector.h
/// \brief Math vector class template of fixed dimension with arithmetic operators and generic
///        functions.
///
/// See \ref mi_math_vector.

#ifndef MI_MATH_VECTOR_H
#define MI_MATH_VECTOR_H

#include <mi/base/types.h>
#include <mi/math/assert.h>
#include <mi/math/function.h>

namespace mi {

namespace math {

/** \ingroup mi_math

    Enum used for initializing a vector from an iterator.

    \see mi::math::Vector

    Two examples initializing a 3D vector:
    \code
    mi::Float32 data[3] = {1.0, 2.0, 4.0};
    mi::math::Vector< mi::Float64, 3> vec( mi::math::FROM_ITERATOR, data);

    std::vector v( 2.4, 3000);
    mi::math::Vector< mi::Float64, 3> vec( mi::math::FROM_ITERATOR, v.begin());
    \endcode
*/
enum From_iterator_tag {
    FROM_ITERATOR ///< Unique enumerator of #From_iterator_tag
};


// Color and Vector can be converted into each other. To avoid cyclic dependencies among the
// headers, the Color_struct class is already defined here.

//------- POD struct that provides storage for color elements --------

/** \ingroup mi_math_color

    Generic storage class template for an RGBA color representation storing four floating points
    elements. Used as %base class for the #mi::math::Color class.

    Use the #mi::math::Color class in your programs and this storage class only if you need a POD
    type, for example for parameter passing.

    All elements are usually in the range [0,1], but they may lie outside that range and they may
    also be negative.

    This class provides storage for four elements of type mi::Float32. These elements can be
    accessed as data members named \c r, \c g, \c b, and \c a. For array-like access of these
    elements, there order in memory is \c rgba.

    This class contains only the data and no member functions. See #mi::math::Color for more.

    \par Include File:
    <tt> \#include <mi/math/color.h></tt>
*/
struct Color_struct
{
    /// Red color component
    Float32 r;
    /// Green color component
    Float32 g;
    /// Blue color component
    Float32 b;
    /// Alpha value, 0.0 is fully transparent and 1.0 is opaque; value can lie outside that range.
    Float32 a;
};


/** \defgroup mi_math_vector Math Vector Class
    \ingroup mi_math

    Math vector class template of fixed dimension with generic operations.

    \par Include File:
    <tt> \#include <mi/math/vector.h></tt>

    @{
*/

/** \defgroup mi_math_vector_struct Internal Storage Class for Math Vector
    \ingroup mi_math_vector

    Storage class for %math vectors with support for \c x, \c y, \c z, and \c w members for
    appropriate dimensions.

    Use the #mi::math::Vector template in your programs and this storage class only if you need a
    POD type, for example for parameter passing.

    \par Include File:
    <tt> \#include <mi/math/vector.h></tt>

    @{
*/

//------- POD struct that provides storage for vector elements --------

/** Generic storage class template for %math vector representations storing \c DIM elements of type
    \c T.

    Used as %base class for the #mi::math::Vector class template.

    Use the #mi::math::Vector template in your programs and this storage class only if you need a
    POD type, for example for parameter passing.

    This class template provides array-like storage for \c DIM many values of a (arithmetic) type
    \c T. It has specializations for \c DIM == 1,2,3, and 4. These specializations use data members
    named \c x, \c y, \c z, and \c w according to the dimension. They provide users of the
    #mi::math::Vector class template with the conventional member access to vector elements, such as
    in:
    \code
    mi::math::Vector<mi::Float64,3> vec( 0.0);
    vec.x = 4.0;
    \endcode

    This class template contains only the data and no member functions. The necessary access
    abstraction is encoded in the free function #mi::math::vector_base_ptr(), which is overloaded
    for the %general case and the various specializations for the small dimensions. It returns a
    pointer to the first element.

    \par Include File:
    <tt> \#include <mi/math/vector.h></tt>
*/
template <typename T, Size DIM>
struct Vector_struct
{
    T elements[DIM]; ///< coordinates.
};

/// Specialization for dimension 1 to create x member
template <typename T> struct Vector_struct<T, 1>
{
    T x;  ///< x-coordinate.
};

/// Specialization for dimension 2 to create x and y member
template <typename T> struct Vector_struct<T, 2>
{
    T x;  ///< x-coordinate.
    T y;  ///< y-coordinate.
};

/// Specialization for dimension 3 to create x, y, and z members
template <typename T> struct Vector_struct<T, 3>
{
    T x;  ///< x-coordinate.
    T y;  ///< y-coordinate.
    T z;  ///< z-coordinate.
};

/// Specialization for dimension 4 to create x, y, z, and w members
template <typename T> struct Vector_struct<T, 4>
{
    T x;  ///< x-coordinate.
    T y;  ///< y-coordinate.
    T z;  ///< z-coordinate.
    T w;  ///< w-coordinate.
};


//------ Indirect access to vector storage base ptr to keep Vector_struct a POD --

/// Returns the %base pointer to the vector data.
template <typename T, Size DIM>
inline T* vector_base_ptr( Vector_struct<T,DIM>& vec)
{
    return vec.elements;
}

/// Returns the %base pointer to the vector data.
template <typename T, Size DIM>
inline const T* vector_base_ptr( const Vector_struct<T,DIM>& vec)
{
    return vec.elements;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==1.
template <typename T>
inline T* vector_base_ptr( Vector_struct<T,1>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==1.
template <typename T>
inline const T* vector_base_ptr( const Vector_struct<T,1>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==2.
template <typename T>
inline T* vector_base_ptr( Vector_struct<T,2>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==2.
template <typename T>
inline const T* vector_base_ptr( const Vector_struct<T,2>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==3.
template <typename T>
inline T* vector_base_ptr( Vector_struct<T,3>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==3.
template <typename T>
inline const T* vector_base_ptr( const Vector_struct<T,3>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==4.
template <typename T>
inline T* vector_base_ptr( Vector_struct<T,4>& vec)
{
    return &vec.x;
}

/// Returns the %base pointer to the vector data, specialization for \c DIM==4.
template <typename T>
inline const T* vector_base_ptr( const Vector_struct<T,4>& vec)
{
    return &vec.x;
}


/*@}*/ // end group mi_math_vector_struct


//------ Generic Vector Class -------------------------------------------------

/** Fixed-size %math vector class template with generic operations.

    This class template provides array-like storage for \c DIM many values of an arithmetic type
    \c T. Several functions and arithmetic operators support the work with vectors.

    An instantiation of the vector class template is a model of the STL container concept. It
    provides random access to its elements and corresponding random access iterators.

    The template parameters have the following requirements:
      - \b T: an arithmetic type supporting <tt>+ - * / == != < > <= >= sqrt() </tt>.
      - \b DIM: a value > 0 of type #mi::Size that defines the fixed dimension of the vector.

    Depending on the dimension \c DIM, the #mi::math::Vector class template offers element access
    through the conventional data members named \c x, \c y, \c z, and \c w. Assuming a vector \c vec
    of suitable dimension, the following expressions are valid
      - \c vec.x; equivalent to \c vec[0] and available if <tt>1 <= DIM <= 4</tt>.
      - \c vec.y; equivalent to \c vec[1] and available if <tt>2 <= DIM <= 4</tt>.
      - \c vec.z; equivalent to \c vec[2] and available if <tt>3 <= DIM <= 4</tt>.
      - \c vec.w; equivalent to \c vec[3] and available if <tt>4 <= DIM <= 4</tt>.

    These data members allow users to access elements, as illustrated in the following example:
    \code
    mi::math::Vector<mi::Float64,3> vec( 0.0);
    vec.x = 4.0;
    \endcode

    \see
        For the free functions and operators available for vectors and vector-like classes see \ref
        mi_math_vector.

   \see
        The underlying POD type #mi::math::Vector_struct.

    \par Include File:
    <tt> \#include <mi/math/vector.h></tt>
*/
template < class T, Size DIM>
class Vector : public Vector_struct<T, DIM> //-V690 PVS
{
public:
    using Pod_type        = Vector_struct<T, DIM>;  ///< POD class corresponding to this vector.
    using storage_type    = Vector_struct<T, DIM>;  ///< Storage class used by this vector.

    using value_type      = T;                      ///< Element type.
    using size_type       = Size;                   ///< Size type, unsigned.
    using difference_type = Difference;             ///< Difference type, signed.
    using pointer         = T*;                     ///< Mutable pointer to element.
    using const_pointer   = const T*;               ///< Const pointer to element.
    using reference       = T&;                     ///< Mutable reference to element.
    using const_reference = const T&;               ///< Const reference to element.

    static constexpr Size DIMENSION = DIM;          ///< Constant dimension of the vector.
    static constexpr Size SIZE      = DIM;          ///< Constant size of the vector.

     /// Constant size of the vector.
    static constexpr inline Size size()     { return SIZE; }

     /// Constant maximum size of the vector.
    static constexpr inline Size max_size() { return SIZE; }

    /// Returns the pointer to the first vector element.
    inline T* begin() { return vector_base_ptr( *this); }

    /// Returns the pointer to the first vector element.
    inline const T* begin() const { return vector_base_ptr( *this); }

    /// Returns the past-the-end pointer.
    ///
    /// The range [begin(),end()) forms the range over all vector elements.
    inline T* end() { return begin() + DIM; }

    /// Returns the past-the-end pointer.
    ///
    /// The range [begin(),end()) forms the range over all vector elements.
    inline const T* end() const { return begin() + DIM; }

    /// The default constructor leaves the vector elements uninitialized.
    inline Vector()
    {
#ifndef NDEBUG
        // In debug mode, default-constructed vectors are initialized with signaling NaNs or, if not
        // applicable, with a maximum value to increase the chances of diagnosing incorrect use of
        // an uninitialized vector.
        using Traits = mi::base::numeric_traits<T>;
        T v = (Traits::has_signaling_NaN) ? Traits::signaling_NaN()
                                          : Traits::max MI_PREVENT_MACRO_EXPAND ();
        for( Size i(0u); i < DIM; ++i)
            (*this)[i] = v;
#endif
    }

    /// Default copy constructor.
    Vector( const Vector<T,DIM>& vec ) = default;

    /// Constructor from underlying storage type.
    inline Vector( const Vector_struct<T,DIM>& vec )
    {
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = vector_base_ptr(vec)[i];
    }

#if (__cplusplus >= 201402L)
private:
    template<Size s>
    constexpr inline void init( std::integral_constant<Size,s>, T v)
    {
        for( Size i = 0; i < DIM; ++i)
            Pod_type::elements[i] = v;
    }

    constexpr inline void init( std::integral_constant<Size,1>, T v)
    { Pod_type::x = v; }

    constexpr inline void init( std::integral_constant<Size,2>, T v)
    { Pod_type::x = Pod_type::y = v; }

    constexpr inline void init( std::integral_constant<Size,3>, T v)
    { Pod_type::x = Pod_type::y = Pod_type::z = v; }

    constexpr inline void init( std::integral_constant<Size,4>, T v)
    { Pod_type::x = Pod_type::y = Pod_type::z = Pod_type::w = v; }

public:
    /// Constructor initializes all vector elements to the value \p v.
    constexpr inline explicit Vector(T v)
    {
        init( std::integral_constant<Size,DIM>(), v);
    }
#else
    inline explicit Vector(T v)
    {
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = v;
    }
#endif

    /** Constructor requires the #mi::math::FROM_ITERATOR tag as first argument and initializes the
        vector elements with the first \c DIM elements from the sequence starting at the iterator
        \p p.

        \c Iterator must be a model of an input iterator. The value type of \c Iterator must be
        assignment compatible with the vector elements type \c T.

        An example:
        \code
        std::vector<int> data( 10, 42); // ten elements of value 42
        mi::math::Vector<mi::Float64, 3> vec( mi::math::FROM_ITERATOR, data.begin());
        \endcode
    */
    template <typename Iterator>
    inline Vector(From_iterator_tag, Iterator p)
    {
        for( Size i(0u); i < DIM; ++i, ++p)
            begin()[i] = *p;
    }

    /** Constructor initializes the vector elements from an \c array of dimension \c DIM.

        The value type \c T2 of the \c array must be assignment compatible with the vector elements
        type \c T.

        An example:
        \code
        int data[3] = { 1, 2, 4};
        mi::math::Vector<mi::Float64, 3> vec( data);
        \endcode
    */
    template <typename T2>
    inline explicit Vector( T2 const (& array)[DIM])
    {
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = array[i];
    }

    /// Template constructor that allows explicit conversions from other vectors with assignment
    /// compatible element value type.
    template <typename T2>
    inline explicit Vector( const Vector<T2,DIM>& other)
    {
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = T(other[i]);
    }

    /// Template constructor that allows explicit conversions from underlying storage type with
    /// assignment compatible element value type.
    template <typename T2>
    inline explicit Vector( const Vector_struct<T2,DIM>& other)
    {
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = T(vector_base_ptr(other)[i]);
    }

    /// Dedicated constructor, for dimension 2 only, that initializes the vector elements from the
    /// two elements \c (v1,v2).
    ///
    /// \pre <tt>DIM == 2</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, T v2) : Vector_struct<T, DIM>{v1,v2}
    {
        mi_static_assert(DIM == 2);
    }

    /// Dedicated constructor, for dimension 3 only, that initializes the vector elements from the
    /// three elements \c (v1,v2,v3).
    ///
    /// \pre <tt>DIM == 3</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, T v2, T v3) : Vector_struct<T, DIM>{v1,v2,v3}
    {
        mi_static_assert(DIM == 3);
    }

    /// Dedicated constructor, for dimension 3 only, that initializes the vector elements from the
    /// three elements \c (v1,v2.x,v2.y).
    ///
    /// \pre <tt>DIM == 3</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, const Vector<T,2>& v2) : Vector_struct<T, DIM>{v1,v2.x,v2.y}
    {
        mi_static_assert(DIM == 3);
    }

    /// Dedicated constructor, for dimension 3 only, that initializes the vector elements from the
    /// three elements \c (v1.x,v1.y,v2).
    ///
    /// \pre <tt>DIM == 3</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(const Vector<T,2>& v1, T v2) : Vector_struct<T, DIM>{v1.x,v1.y,v2}
    {
        mi_static_assert(DIM == 3);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1,v2,v3,v4).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, T v2, T v3, T v4) : Vector_struct<T, DIM>{v1,v2,v3,v4}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1,v2,v3.x,v3.y).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, T v2, const Vector<T,2>& v3) : Vector_struct<T, DIM>{v1,v2,v3.x,v3.y}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1,v2.x,v2.y,v3).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, const Vector<T,2>& v2, T v3) : Vector_struct<T, DIM>{v1,v2.x,v2.y,v3}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1.x,v1.y,v2,v3).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(const Vector<T,2>& v1, T v2, T v3) : Vector_struct<T, DIM>{v1.x,v1.y,v2,v3}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1.x,v1.y,v2.x,v2.y).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(const Vector<T,2>& v1, const Vector<T,2>& v2) : Vector_struct<T, DIM>{v1.x,v1.y,v2.x,v2.y}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1,v2.x,v2.y,v2.z).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(T v1, const Vector<T,3>& v2) : Vector_struct<T, DIM>{v1,v2.x,v2.y,v2.z}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from the
    /// four elements \c (v1.x,v1.y,v1.z,v2).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector(const Vector<T,3>& v1, T v2) : Vector_struct<T, DIM>{v1.x,v1.y,v1.z,v2}
    {
        mi_static_assert(DIM == 4);
    }

    /// Dedicated constructor, for dimension 4 only, that initializes the vector elements from a
    /// color interpreted as a vector (r,g,b,a).
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline explicit Vector( const Color_struct& color) : Vector_struct<T, DIM>{static_cast<T>(color.r),static_cast<T>(color.g),static_cast<T>(color.b),static_cast<T>(color.a)}
    {
        mi_static_assert(DIM == 4);
    }

    /// Assignment.
    Vector& operator= ( const Vector& other) = default;

    /// Assignment from a scalar, setting all elements to \p s.
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector& operator= ( T s)
    {
#if (__cplusplus >= 201402L)
        init(std::integral_constant<Size,DIM>(), s);
#else
        for( Size i(0u); i < DIM; ++i)
            begin()[i] = s;
#endif
        return *this;
    }
    /// Assignment, for dimension 4 only, that assigns color interpreted as a vector (r,g,b,a) to
    /// this vector.
    ///
    /// \pre <tt>DIM == 4</tt>
#if (__cplusplus >= 201402L)
    constexpr
#endif
    inline Vector& operator= ( const Color_struct& color)
    {
        mi_static_assert(DIM == 4);
        Pod_type::x = color.r;
        Pod_type::y = color.g;
        Pod_type::z = color.b;
        Pod_type::w = color.a;
        return *this;
    }

    /// Accesses the \c i-th vector element.
    ///
    /// \pre 0 <= \c i < #size()
    inline T& operator[] (Size i)
    {
        mi_math_assert_msg(i < DIM, "precondition");
        return begin()[i];
    }

    /// Accesses the \c i-th vector element.
    ///
    /// \pre 0 <= \c i < #size()
    inline const T& operator[] (Size i) const
    {
        mi_math_assert_msg(i < DIM, "precondition");
        return begin()[i];
    }

    /// Returns the \c i-th vector element.
    ///
    /// \pre 0 <= \c i < #size()
    inline const T& get(Size i) const
    {
        mi_math_assert_msg(i < DIM, "precondition");
        return begin()[i];
    }

    /// Sets the \c i-th vector element to \c value.
    ///
    /// \pre 0 <= \c i < #size()
    inline void set(Size i, T value)
    {
        mi_math_assert_msg(i < DIM, "precondition");
        begin()[i] = value;
    }



    /// Normalizes this vector to unit length.
    ///
    /// Returns \c false if normalization fails because the vector norm is zero, and \c true
    /// otherwise. This vector remains unchanged if normalization failed.
    ///
    /// Uses an unqualified call to \c sqrt(...) on the vector element type.
    inline bool normalize()
    {
        const T rec_length = T(1) / length( *this);
        const bool result = isfinite( rec_length); //-V601 PVS
        if( result)
            (*this) *= rec_length;
        return result;
    }


    //------ Free comparison operators ==, !=, <, <=, >, >= for vectors --------

    /// Returns \c true if \c lhs is elementwise equal to \c rhs.
    inline bool operator==( Vector<T,DIM> rhs) const
    {
        return is_equal( *this, rhs);
    }

    /// Returns \c true if \c lhs is elementwise not equal to \c rhs.
    inline bool operator!=( Vector<T,DIM> rhs) const
    {
        return is_not_equal( *this, rhs);
    }

    /// Returns \c true if \c lhs is lexicographically less than \c rhs.
    ///
    /// \see   \ref mi_def_lexicographic_order
    inline bool operator<( Vector<T,DIM> rhs) const
    {
        return lexicographically_less( *this, rhs);
    }

    /// Returns \c true if \c lhs is lexicographically less than or equal to \c rhs.
    ///
    /// \see   \ref mi_def_lexicographic_order
    inline bool operator<=( Vector<T,DIM> rhs) const
    {
        return lexicographically_less_or_equal( *this, rhs);
    }

    /// Returns \c true if \c lhs is lexicographically greater than \c rhs.
    ///
    /// \see   \ref mi_def_lexicographic_order
    inline bool operator>( Vector<T,DIM> rhs) const
    {
        return lexicographically_greater( *this, rhs);
    }

    /// Returns \c true if \c lhs is lexicographically greater than or equal to \c rhs.
    ///
    /// \see   \ref mi_def_lexicographic_order
    inline bool operator>=( Vector<T,DIM> rhs) const
    {
        return lexicographically_greater_or_equal( *this, rhs);
    }
};


//------ Free operators +=, -=, *=, /=, +, -, *, and / for vectors -------------

/// Adds \p rhs elementwise to \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator+=(
    Vector<T,DIM>&              lhs,
    const Vector_struct<T,DIM>& rhs)
{
    for( Size i(0u); i < DIM; ++i)
        lhs.elements[i] += rhs.elements[i];
    return lhs;
}
template <typename T>
inline Vector<T,1>& operator+=(
    Vector<T,1>&              lhs,
    const Vector_struct<T,1>& rhs)
{
    lhs.x += rhs.x;
    return lhs;
}
template <typename T>
inline Vector<T,2>& operator+=(
    Vector<T,2>&              lhs,
    const Vector_struct<T,2>& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
template <typename T>
inline Vector<T,3>& operator+=(
    Vector<T,3>&              lhs,
    const Vector_struct<T,3>& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}
template <typename T>
inline Vector<T,4>& operator+=(
    Vector<T,4>&              lhs,
    const Vector_struct<T,4>& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

/// Subtracts \p rhs elementwise from \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator-=(
    Vector<T,DIM>&              lhs,
    const Vector_struct<T,DIM>& rhs)
{
    for( Size i(0u); i < DIM; ++i)
        lhs.elements[i] -= rhs.elements[i];
    return lhs;
}
template <typename T>
inline Vector<T,1>& operator-=(
    Vector<T,1>&              lhs,
    const Vector_struct<T,1>& rhs)
{
    lhs.x -= rhs.x;
    return lhs;
}
template <typename T>
inline Vector<T,2>& operator-=(
    Vector<T,2>&              lhs,
    const Vector_struct<T,2>& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}
template <typename T>
inline Vector<T,3>& operator-=(
    Vector<T,3>&              lhs,
    const Vector_struct<T,3>& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}
template <typename T>
inline Vector<T,4>& operator-=(
    Vector<T,4>&              lhs,
    const Vector_struct<T,4>& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}

/// Multiplies \p rhs elementwise with \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator*=(
    Vector<T,DIM>&              lhs,
    const Vector_struct<T,DIM>& rhs)
{
    for( Size i(0u); i < DIM; ++i)
        lhs.elements[i] *= rhs.elements[i];
    return lhs;
}
template <typename T>
inline Vector<T,1>& operator*=(
    Vector<T,1>&              lhs,
    const Vector_struct<T,1>& rhs)
{
    lhs.x *= rhs.x;
    return lhs;
}
template <typename T>
inline Vector<T,2>& operator*=(
    Vector<T,2>&              lhs,
    const Vector_struct<T,2>& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}
template <typename T>
inline Vector<T,3>& operator*=(
    Vector<T,3>&              lhs,
    const Vector_struct<T,3>& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}
template <typename T>
inline Vector<T,4>& operator*=(
    Vector<T,4>&              lhs,
    const Vector_struct<T,4>& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    lhs.w *= rhs.w;
    return lhs;
}

/// Computes \p lhs modulo \p rhs elementwise and returns the modified \p lhs.
/// Only defined for typenames \p T having the % operator.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator%=(
    Vector<T,DIM>&              lhs,
    const Vector_struct<T,DIM>& rhs)
{
    for( Size i(0u); i < DIM; ++i)
        lhs[i] %= vector_base_ptr(rhs)[i];
    return lhs;
}

/// Divides \p lhs elementwise by \p rhs and returns the modified \p lhs.
template <typename T, typename U, Size DIM>
inline Vector<T,DIM>& operator/=(
    Vector<T,DIM>&              lhs,
    const Vector_struct<U,DIM>& rhs)
{
    for( Size i(0u); i < DIM; ++i)
        lhs[i] = T(lhs[i] / vector_base_ptr(rhs)[i]);
    return lhs;
}

/// Adds \p lhs and \p rhs elementwise and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator+(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> tmp( lhs);
    return tmp += rhs;
}

/// Subtracts \p rhs elementwise from \p lhs and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator-(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> tmp( lhs);
    return tmp -= rhs;
}

/// Multiplies \p rhs elementwise with \p lhs and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator*(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> tmp( lhs);
    return tmp *= rhs;
}

/// Computes \p lhs modulo \p rhs elementwise and returns the new result.
/// Only defined for typenames \p T having the % operator.
template <typename T, Size DIM>
inline Vector<T,DIM> operator%(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> tmp( lhs);
    return tmp %= rhs;
}

/// Divides \p rhs elementwise by \p lhs and returns the new result.
template <typename T, typename U, Size DIM>
inline Vector<T,DIM> operator/(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<U,DIM>& rhs)
{
    Vector<T,DIM> tmp(lhs);
    return tmp /= rhs;
}

/// Negates the vector \p v elementwise and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator-( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> tmp;
    for( Size i(0u); i < DIM; ++i)
        tmp.elements[i] = -v.elements[i];
    return tmp;
}
template <typename T>
inline Vector<T,1> operator-( const Vector_struct<T,1>& v)
{
    return Vector<T,1>( -v.x);
}
template <typename T>
inline Vector<T,2> operator-( const Vector_struct<T,2>& v)
{
    return Vector<T,2>( -v.x, -v.y);
}
template <typename T>
inline Vector<T,3> operator-( const Vector_struct<T,3>& v)
{
    return Vector<T,3>( -v.x, -v.y, -v.z);
}
template <typename T>
inline Vector<T,4> operator-( const Vector_struct<T,4>& v)
{
    return Vector<T,4>( -v.x, -v.y, -v.z, -v.w);
}


//------ Resolve ambiguity with hybrid operators -------------
// (Those operators should really be restricted to scalar types but that requires
// modern C++ which some users of this headers don't enable.)

/// Adds \p rhs elementwise to \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator+=(
    Vector<T,DIM>&       lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs += static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Subtracts \p rhs elementwise from \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator-=(
    Vector<T,DIM>&       lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs -= static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Multiplies \p rhs elementwise with \p lhs and returns the modified \p lhs.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator*=(
    Vector<T,DIM>&       lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs *= static_cast<const Vector_struct<T,DIM>&>(rhs);
}
template <typename T>
inline Vector<T,1>& operator*=(
    Vector<T,1>&       lhs,
    const Vector<T,1>& rhs)
{
    return lhs *= static_cast<const Vector_struct<T,1>&>(rhs);
}
template <typename T>
inline Vector<T,2>& operator*=(
    Vector<T,2>&       lhs,
    const Vector<T,2>& rhs)
{
    return lhs *= static_cast<const Vector_struct<T,2>&>(rhs);
}
template <typename T>
inline Vector<T,3>& operator*=(
    Vector<T,3>&       lhs,
    const Vector<T,3>& rhs)
{
    return lhs *= static_cast<const Vector_struct<T,3>&>(rhs);
}
template <typename T>
inline Vector<T,4>& operator*=(
    Vector<T,4>&       lhs,
    const Vector<T,4>& rhs)
{
    return lhs *= static_cast<const Vector_struct<T,4>&>(rhs);
}

/// Computes \p lhs modulo \p rhs elementwise and returns the modified \p lhs.
/// Only defined for typenames \p T having the % operator.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator%=(
    Vector<T,DIM>&       lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs %= static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Divides \p lhs elementwise by \p rhs and returns the modified \p lhs.
template <typename T, typename U, Size DIM>
inline Vector<T,DIM>& operator/=(
    Vector<T,DIM>&       lhs,
    const Vector<U,DIM>& rhs)
{
    return lhs /= static_cast<const Vector_struct<U,DIM>&>(rhs);
}

/// Adds \p lhs and \p rhs elementwise and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator+(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs + static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Subtracts \p rhs elementwise from \p lhs and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator-(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs - static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Multiplies \p rhs elementwise with \p lhs and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator*(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs * static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Computes \p lhs modulo \p rhs elementwise and returns the new result.
/// Only defined for typenames \p T having the % operator.
template <typename T, Size DIM>
inline Vector<T,DIM> operator%(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    return lhs % static_cast<const Vector_struct<T,DIM>&>(rhs);
}

/// Divides \p rhs elementwise by \p lhs and returns the new result.
template <typename T, typename U, Size DIM>
inline Vector<T,DIM> operator/(
    const Vector<T,DIM>& lhs,
    const Vector<U,DIM>& rhs)
{
    return lhs / static_cast<const Vector_struct<U,DIM>&>(rhs);
}

/// Negates the vector \p v elementwise and returns the new result.
template <typename T, Size DIM>
inline Vector<T,DIM> operator-( const Vector<T,DIM>& v)
{
    return -static_cast<const Vector_struct<T,DIM>&>(v);
}


//------ Free operator *=, /=, *, and / definitions for scalars ---------------

/// Multiplies the vector \p v elementwise with the scalar \p s and returns the modified vector
/// \p v.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM>& operator*=(
    Vector<T,DIM>& v,
    TT             s)
{
    for( Size i(0u); i < DIM; ++i)
        v.elements[i] = T(v.elements[i] * s);
    return v;
}
template <typename T, typename TT>
inline Vector<T,1>& operator*=(
    Vector<T,1>& v,
    TT           s)
{
    v.x = T(v.x * s);
    return v;
}
template <typename T, typename TT>
inline Vector<T,2>& operator*=(
    Vector<T,2>& v,
    TT           s)
{
    v.x = T(v.x * s);
    v.y = T(v.y * s);
    return v;
}
template <typename T, typename TT>
inline Vector<T,3>& operator*=(
    Vector<T,3>& v,
    TT           s)
{
    v.x = T(v.x * s);
    v.y = T(v.y * s);
    v.z = T(v.z * s);
    return v;
}
template <typename T, typename TT>
inline Vector<T,4>& operator*=(
    Vector<T,4>& v,
    TT           s)
{
    v.x = T(v.x * s);
    v.y = T(v.y * s);
    v.z = T(v.z * s);
    v.w = T(v.w * s);
    return v;
}

/// Computes \p v modulo \p s elementwise and returns the modified vector \p v.
///
/// Only defined for typenames \p T having the % operator for \p TT arguments.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM>& operator%=(
    Vector<T,DIM>& v,
    TT             s)
{
    for( Size i(0u); i < DIM; ++i)
        v[i] = T(v[i] % s);
    return v;
}

/// Divides the vector \p v elementwise by the scalar \p s and returns the modified vector \p v.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM>& operator/=(
    Vector<T,DIM>&  v,
    TT              s)
{
    for( Size i(0u); i < DIM; ++i)
        v[i] = T(v[i] / s);
    return v;
}

/// Multiplies the vector \p v elementwise with the scalar \p s and returns the new result.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM> operator*(
    const Vector_struct<T,DIM>& v,
    TT                          s)
{
    Vector<T,DIM> tmp( v);
    return tmp *= s;
}

/// Multiplies the vector \p v elementwise with the scalar \p s and returns the new result.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM> operator*(
    TT                          s,
    const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> tmp( v);
    return tmp *= s;
}

/// Computes \p v modulo \p s elementwise and returns the new result.
///
/// Only defined for typenames \p T having the % operator for \p TT arguments.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM> operator%(
    const Vector_struct<T,DIM>& v,
    TT                          s)
{
    Vector<T,DIM> tmp( v);
    return tmp %= s;
}

/// Divides the vector \p v elementwise by the scalar \p s and returns the new result.
template <typename T, typename TT, Size DIM>
inline Vector<T,DIM> operator/(
    const Vector_struct<T,DIM>& v,
    TT                          s)
{
    Vector<T,DIM> tmp( v);
    return tmp /= s;
}


//------ Free operator ++, -- for vectors -------------------------------------

/// Pre-increments all elements of \p vec and returns the result. Modifies \p vec.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator++( Vector<T,DIM>& vec)
{
    general::for_each( vec, functor::Operator_pre_incr());
    return vec;
}

/// Pre-decrements all elements of \p vec and returns the result. Modifies \p vec.
template <typename T, Size DIM>
inline Vector<T,DIM>& operator--( Vector<T,DIM>& vec)
{
    general::for_each( vec, functor::Operator_pre_decr());
    return vec;
}


//------ Free operators !, &&, ||, ^ for bool vectors and bool scalars ---------

/// Returns the elementwise logical and of two boolean vectors.
template <Size DIM>
inline Vector<bool,DIM> operator&&(
    const Vector<bool,DIM>& lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result, functor::Operator_and_and());
    return result;
}

/// Returns the elementwise logical and of a bool and a boolean vector.
template <Size DIM>
inline Vector<bool,DIM> operator&&(
    bool                    lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform_left_scalar( lhs, rhs, result, functor::Operator_and_and());
    return result;
}

/// Returns the elementwise logical and of a boolean vector and a bool.
template <Size DIM>
inline Vector<bool,DIM> operator&&(
    const Vector<bool,DIM>& lhs,
    bool                    rhs)
{
    Vector<bool,DIM> result;
    general::transform_right_scalar( lhs, rhs, result, functor::Operator_and_and());
    return result;
}

/// Returns the elementwise logical or of two boolean vectors.
template <Size DIM>
inline Vector<bool,DIM> operator||(
    const Vector<bool,DIM>& lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform(lhs, rhs, result, functor::Operator_or_or());
    return result;
}

/// Returns the elementwise logical or of a bool and a boolean vector.
template <Size DIM>
inline Vector<bool,DIM> operator||(
    bool                    lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform_left_scalar( lhs, rhs, result, functor::Operator_or_or());
    return result;
}

/// Returns the elementwise logical or of a boolean vector and a bool.
template <Size DIM>
inline Vector<bool,DIM> operator||(
    const Vector<bool,DIM>& lhs,
    bool                     rhs)
{
    Vector<bool,DIM> result;
    general::transform_right_scalar( lhs, rhs, result, functor::Operator_or_or());
    return result;
}

/// Returns the elementwise logical xor of two boolean vectors.
template <Size DIM>
inline Vector<bool,DIM> operator^(
    const Vector<bool,DIM>& lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result, functor::Operator_xor());
    return result;
}

/// Returns the elementwise logical xor of a bool and a boolean vector.
template <Size DIM>
inline Vector<bool,DIM> operator^(
    bool                    lhs,
    const Vector<bool,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform_left_scalar( lhs, rhs, result, functor::Operator_xor());
    return result;
}

/// Returns the elementwise logical xor of a boolean vector and a bool.
template <Size DIM>
inline Vector<bool,DIM> operator^(
    const Vector<bool,DIM>& lhs,
    bool                    rhs)
{
    Vector<bool,DIM> result;
    general::transform_right_scalar( lhs, rhs, result, functor::Operator_xor());
    return result;
}

/// Returns the elementwise logical not of a boolean vector.
template <Size DIM>
inline Vector<bool,DIM> operator!(
    const Vector<bool,DIM>& vec)
{
    Vector<bool,DIM> result;
    general::transform( vec, result, functor::Operator_not());
    return result;
}


//------ Elementwise comparison operators returning a bool vector. ------------

/// Returns the boolean vector result of an elementwise equality comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_equal(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_equal_equal());
    return result;
}

/// Returns the boolean vector result of an elementwise inequality comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_not_equal(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_not_equal());
    return result;
}

/// Returns the boolean vector result of an elementwise less-than comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_less_than(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_less());
    return result;
}

/// Returns the boolean vector result of an elementwise less-than-or-equal comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_less_than_or_equal(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_less_equal());
    return result;
}

/// Returns the boolean vector result of an elementwise greater-than comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_greater_than(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_greater());
    return result;
}

/// Returns the boolean vector result of an elementwise greater-than-or-equal comparison.
template <typename T, Size DIM>
inline Vector<bool,DIM> elementwise_is_greater_than_or_equal(
    const Vector<T,DIM>& lhs,
    const Vector<T,DIM>& rhs)
{
    Vector<bool,DIM> result;
    general::transform( lhs, rhs, result,functor::Operator_greater_equal());
    return result;
}


//------ Function Overloads for Vector Algorithms -----------------------------

/// Returns a vector with the elementwise absolute values of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> abs( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = abs( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise arc cosine of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> acos( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = acos( vector_base_ptr(v)[i]);
    return result;
}

/// Returns \c true if \c all of all elements of \c v returns \c true.
template <typename T, Size DIM>
inline bool all( const Vector_struct<T,DIM>& v)
{
    for( Size i = 0; i != DIM; ++i)
        if( !all( vector_base_ptr(v)[i]))
            return false;
    return true;
}

/// Returns \c true if \c any of any element of \c v returns \c true.
template <typename T, Size DIM>
inline bool any( const Vector_struct<T,DIM>& v)
{
    for( Size i = 0; i != DIM; ++i)
        if( any( vector_base_ptr(v)[i]))
           return true;
    return false;
}

/// Returns a vector with the elementwise arc sine of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> asin( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = asin( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise arc tangent of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> atan( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = atan( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise arc tangent of the vector \p v / \p w.
///
/// The signs of the elements of \p v and \p w are used to determine the quadrant of the results.
template <typename T, Size DIM>
inline Vector<T,DIM> atan2( const Vector_struct<T,DIM>& v, const Vector_struct<T,DIM>& w)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = atan2( vector_base_ptr(v)[i], vector_base_ptr(w)[i]);
    return result;
}

/// Returns a vector with the elementwise smallest integral value that is not less than the element
/// in vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> ceil( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = ceil( vector_base_ptr(v)[i]);
    return result;
}

/// Returns the vector \p v elementwise clamped to the range [\p low, \p high].
template <typename T, Size DIM>
inline Vector<T,DIM> clamp(
    const Vector_struct<T,DIM>&  v,
    const Vector_struct<T,DIM>&  low,
    const Vector_struct<T,DIM>&  high)
{
    Vector<T,DIM> result;
    for( Size i = 0u; i < DIM; ++i)
        result[i] = clamp(
            vector_base_ptr(v)[i], vector_base_ptr(low)[i], vector_base_ptr(high)[i]);
    return result;
}

/// Returns the vector \p v elementwise clamped to the range [\p low, \p high].
template <typename T, Size DIM>
inline Vector<T,DIM> clamp(
    const Vector_struct<T,DIM>& v,
    const Vector_struct<T,DIM>& low,
    T                    high)
{
    Vector<T,DIM> result;
    for( Size i = 0u; i < DIM; ++i)
        result[i] = clamp( vector_base_ptr(v)[i], vector_base_ptr(low)[i], high);
    return result;
}

/// Returns the vector \p v elementwise clamped to the range [\p low, \p high].
template <typename T, Size DIM>
inline Vector<T,DIM> clamp(
    const Vector_struct<T,DIM>& v,
    T                    low,
    const Vector_struct<T,DIM>& high)
{
    Vector<T,DIM> result;
    for( Size i = 0u; i < DIM; ++i)
        result[i] = clamp( vector_base_ptr(v)[i], low, vector_base_ptr(high)[i]);
    return result;
}

/// Returns the vector \p v elementwise clamped to the range [\p low, \p high].
template <typename T, Size DIM>
inline Vector<T,DIM> clamp(
    const Vector_struct<T,DIM>& v,
    T                    low,
    T                    high)
{
    Vector<T,DIM> result;
    for( Size i = 0u; i < DIM; ++i)
        result[i] = clamp( vector_base_ptr(v)[i], low, high);
    return result;
}

/// Returns a vector with the elementwise cosine of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> cos( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = cos( vector_base_ptr(v)[i]);
    return result;
}

/// Converts elementwise radians in \p v to degrees.
template <typename T, Size DIM>
inline Vector<T,DIM> degrees( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = degrees( vector_base_ptr(v)[i]);
    return result;
}

/// Returns elementwise maximum of two vectors.
template <typename T, Size DIM>
inline Vector<T,DIM> elementwise_max(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> r;
    for( Size i(0u); i < Vector<T,DIM>::DIMENSION; ++i)
        r[i] = base::max MI_PREVENT_MACRO_EXPAND (
            vector_base_ptr(lhs)[i], vector_base_ptr(rhs)[i] );
    return r;
}

/// Returns elementwise minimum of two vectors.
template <typename T, Size DIM>
inline Vector<T,DIM> elementwise_min(
    const Vector_struct<T,DIM>& lhs,
    const Vector_struct<T,DIM>& rhs)
{
    Vector<T,DIM> r;
    for( Size i(0u); i < Vector<T,DIM>::DIMENSION; ++i)
        r[i] = base::min MI_PREVENT_MACRO_EXPAND (
            vector_base_ptr(lhs)[i], vector_base_ptr(rhs)[i] );
    return r;
}

/// Returns a vector with elementwise \c e to the power of the element in the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> exp( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = exp( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with elementwise \c 2 to the power of the element in the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> exp2( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = exp2( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise largest integral value that is not greater than the
/// element in vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> floor( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = floor( vector_base_ptr(v)[i]);
    return result;
}

/// Returns elementwise \p a modulo \p b, in other words, the remainder of a/b.
///
/// The elementwise result has the same sign as \p a.
template <typename T, Size DIM>
inline Vector<T,DIM> fmod( const Vector_struct<T,DIM>& a, const Vector_struct<T,DIM>& b)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = fmod( vector_base_ptr(a)[i], vector_base_ptr(b)[i]);
    return result;
}

/// Returns elementwise \p a modulo \p b, in other words, the remainder of a/b.
///
/// The elementwise result has the same sign as \p a.
template <typename T, Size DIM>
inline Vector<T,DIM> fmod( const Vector_struct<T,DIM>& a, T b)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = fmod( vector_base_ptr(a)[i], b);
    return result;
}

/// Returns a vector with the elementwise positive fractional part of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> frac( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = frac( vector_base_ptr(v)[i]);
    return result;
}

/// Compares the two given values elementwise for equality within the given epsilon.
template <typename T, Size DIM>
inline bool is_approx_equal(
    const Vector_struct<T,DIM>& left,
    const Vector_struct<T,DIM>& right,
    T                    e)
{
    for( Size i = 0u; i < DIM; ++i)
        if( !is_approx_equal( vector_base_ptr(left)[i], vector_base_ptr(right)[i], e))
            return false;
    return true;
}

/// Returns the elementwise linear interpolation between \p v1 and \c v2, i.e., it returns
/// <tt>(1-t) * v1 + t * v2</tt>.
template <typename T, Size DIM>
inline Vector<T,DIM> lerp(
    const Vector_struct<T,DIM>& v1,  ///< one vector
    const Vector_struct<T,DIM>& v2,  ///< second vector
    const Vector_struct<T,DIM>& t)   ///< interpolation parameter in [0,1]
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = vector_base_ptr(v1)[i] * (T(1)-vector_base_ptr(t)[i])
                  + vector_base_ptr(v2)[i] * vector_base_ptr(t)[i];
    return result;
}

/// Returns the linear interpolation between \p v1 and \c v2, i.e., it returns
/// <tt>(1-t) * v1 + t * v2</tt>.
template <typename T, Size DIM>
inline Vector<T,DIM> lerp(
    const Vector_struct<T,DIM>& v1,  ///< one vector
    const Vector_struct<T,DIM>& v2,  ///< second vector
    T          t)             ///< interpolation parameter in [0,1]
{
    // equivalent to: return v1 * (T(1)-t) + v2 * t;
    Vector<T,DIM> result;
    T t2 = T(1) - t;
    for( Size i = 0; i != DIM; ++i)
        result[i] = vector_base_ptr(v1)[i] * t2 + vector_base_ptr(v2)[i] * t;
    return result;
}

/// Returns a vector with the elementwise natural logarithm of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> log( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = log( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise %base 2 logarithm of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> log2 MI_PREVENT_MACRO_EXPAND ( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = log2 MI_PREVENT_MACRO_EXPAND ( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise %base 10 logarithm of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> log10( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = log10( vector_base_ptr(v)[i]);
    return result;
}

/// Returns the elementwise fractional part of \p v and stores the elementwise integral part of \p v
/// in \p i.
///
/// Both parts have elementwise the same sign as \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> modf( const Vector_struct<T,DIM>& v, Vector<T,DIM>& i)
{
    Vector<T,DIM> result;
    for( Size j = 0; j != DIM; ++j)
        result[j] = modf( vector_base_ptr(v)[j], i[j]);
    return result;
}

/// Returns the vector \p a  elementwise to the power of \p b.
template <typename T, Size DIM>
inline Vector<T,DIM> pow( const Vector_struct<T,DIM>& a, const Vector_struct<T,DIM>& b)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = pow( vector_base_ptr(a)[i], vector_base_ptr(b)[i]);
    return result;
}

/// Returns the vector \p a  elementwise to the power of \p b.
template <typename T, Size DIM>
inline Vector<T,DIM> pow( const Vector_struct<T,DIM>& a, T b)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = pow( vector_base_ptr(a)[i], b);
    return result;
}

/// Converts elementwise degrees in \p v to radians.
template <typename T, Size DIM>
inline Vector<T,DIM> radians( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = radians( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elements of vector \p v rounded to nearest integers.
template <typename T, Size DIM>
inline Vector<T,DIM> round( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = round( vector_base_ptr(v)[i]);
    return result;
}

/// Returns the reciprocal of the square root of each element of \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> rsqrt( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = rsqrt( vector_base_ptr(v)[i]);
    return result;
}

/// Returns the vector \p v clamped elementwise to the range [0,1].
template <typename T, Size DIM>
inline Vector<T,DIM> saturate( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = saturate( vector_base_ptr(v)[i]);
    return result;
}

/// Returns the elementwise sign of vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> sign( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = sign( vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise sine of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> sin( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = sin( vector_base_ptr(v)[i]);
    return result;
}

/// Computes elementwise the sine \p s and cosine \p c of angles \p a simultaneously.
///
/// The angles \p a are specified in radians.
template <typename T, Size DIM>
inline void sincos( const Vector_struct<T,DIM>& a, Vector<T,DIM>& s, Vector<T,DIM>& c)
{
    for( Size i = 0; i != DIM; ++i)
        sincos( vector_base_ptr(a)[i], s[i], c[i]);
}

/// Returns 0 if \p v is less than \p a and 1 if \p v is greater than \p b in an elementwise
/// fashion.
///
/// A smooth curve is applied in-between so that the return values vary continuously from 0 to 1 as
/// elements in \p v vary from \p a to \p b.
template <typename T, Size DIM>
inline Vector<T,DIM> smoothstep(
    const Vector_struct<T,DIM>& a,
    const Vector_struct<T,DIM>& b,
    const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = smoothstep(
            vector_base_ptr(a)[i], vector_base_ptr(b)[i], vector_base_ptr(v)[i]);
    return result;
}

/// Returns 0 if \p x is less than \p a and 1 if \p x is greater than \p b in an elementwise
/// fashion.
///
/// A smooth curve is applied in-between so that the return values vary continuously from 0 to 1 as
/// \p x varies from \p a to \p b.
template <typename T, Size DIM>
inline Vector<T,DIM> smoothstep(
    const Vector_struct<T,DIM>& a,
    const Vector_struct<T,DIM>& b,
    T x)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = smoothstep( vector_base_ptr(a)[i], vector_base_ptr(b)[i], x);
    return result;
}

/// Returns the square root of each element of \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> sqrt( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = sqrt( vector_base_ptr(v)[i]);
    return result;
}

/// Returns elementwise 0 if \p v is less than \p a and 1 otherwise.
template <typename T, Size DIM>
inline Vector<T,DIM> step( const Vector_struct<T,DIM>& a, const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = step( vector_base_ptr(a)[i], vector_base_ptr(v)[i]);
    return result;
}

/// Returns a vector with the elementwise tangent of the vector \p v.
template <typename T, Size DIM>
inline Vector<T,DIM> tan( const Vector_struct<T,DIM>& v)
{
    Vector<T,DIM> result;
    for( Size i = 0; i != DIM; ++i)
        result[i] = tan( vector_base_ptr(v)[i]);
    return result;
}


//------ Geometric Vector Algorithms ------------------------------------------

/// Returns the two-times-two determinant result for the two vectors \p lhs and \p rhs.
template <typename T>
inline T cross(
    const Vector_struct<T,2>& lhs,
    const Vector_struct<T,2>& rhs)
{
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

/// Returns the three-dimensional cross product result for the two vectors \p lhs and \p rhs.
template <typename T>
inline Vector<T,3> cross(
    const Vector_struct<T,3>& lhs,
    const Vector_struct<T,3>& rhs)
{
    return Vector<T,3>( lhs.y * rhs.z - lhs.z * rhs.y,
                        lhs.z * rhs.x - lhs.x * rhs.z,
                        lhs.x * rhs.y - lhs.y * rhs.x);
}

/// Computes a basis of 3D space with one given vector.
///
/// Given a unit length vector \p n, computes two vectors \p u and \p v such that (\p u, \p n, \p v)
/// forms an orthonormal basis (\p u and \p v are unit length). This function is not continuous with
/// respect to \p n: in some cases, a small perturbation on \p n will flip the basis.
template <typename T>
inline void make_basis(
    const Vector<T,3>& n,  ///< input, normal vector
    Vector<T,3>* const u,  ///< output, first vector in tangent plane
    Vector<T,3>* const v)  ///< output, second vector in tangent plane
{
#ifdef mi_base_assert_enabled
    constexpr T eps = 1e-6f;       // smallest resolvable factor
#endif

    mi_math_assert_msg( u != 0, "precondition");
    mi_math_assert_msg( v != 0, "precondition");
    // Sanity check: the normal vector must be unit length.
    mi_math_assert_msg( abs( length(n) - T(1)) < eps, "precondition");

    // Compute u.
    if( abs(n.x) < abs(n.y)) {
        // u = cross(x, n), x = (1, 0, 0)
        u->x = T(0);
        u->y = -n.z;
        u->z =  n.y;
    } else {
        // u = cross(y, n), y = (0, 1, 0)
        u->x =  n.z;
        u->y = T(0);
        u->z = -n.x;
    }
    u->normalize();

    // Compute v. Since *u and n are orthogonal and unit-length,
    // there is no need to normalize *v.
    *v = cross( *u, n);

    // Sanity check: make sure (u, n, v) is an orthogonal basis.
    mi_math_assert_msg( abs( dot( *u,  n)) < eps, "postcondition");
    mi_math_assert_msg( abs( dot( *u, *v)) < eps, "postcondition");
    mi_math_assert_msg( abs( dot(  n, *v)) < eps, "postcondition");
    // Sanity check: make sure u and v are unit length.
    mi_math_assert_msg( abs( length( *u) - T(1)) < eps, "postcondition");
    mi_math_assert_msg( abs( length( *v) - T(1)) < eps, "postcondition");
}

/// Computes a basis of 3D space with one given vector, plane, and direction.
///
/// Given a unit length vector n, and two non-colinear and non-zero vectors u and v, this function
/// computes two vectors t and b such that (t, n, b) forms an orthonormal basis (t and b are unit
/// length), t lies in the plane formed by n and u, and b has the same orientation as v, i.e.,
///  dot(b, v) >= 0.
template <typename T>
inline void make_basis(
    const Vector<T,3>& n,     ///< input, normal vector
    const Vector<T,3>& u,     ///< input, first direction vector
    const Vector<T,3>& v,     ///< input, second direction vector
    Vector<T,3>* const t,     ///< output, first vector in tangent plane
    Vector<T,3>* const b)     ///< output, second vector in tangent plane
{
#ifdef mi_base_assert_enabled
    constexpr T eps = 1e-6f;       // smallest resolvable factor
#endif

    mi_math_assert_msg( t != 0, "precondition");
    mi_math_assert_msg( b != 0, "precondition");
    // Sanity check: the normal vector must be unit length.
    mi_math_assert_msg( abs( length( n) - T(1)) < eps, "precondition");
    // Sanity check: the other vector lengths should be finite and non-zero
    mi_math_assert_msg( length( u) > 0, "precondition");
    mi_math_assert_msg( length( v) > 0, "precondition");
    mi_math_assert_msg( isfinite( length( u)), "precondition");
    mi_math_assert_msg( isfinite( length( v)), "precondition");

    // Compute b
    *b = cross(u,n);
    b->normalize();

    // Compute t. Since *b and n are orthogonal and unit-length,
    // there is no need to normalize *t.
    *t = cross(n,*b);

    // Check that b has the same orientation of v
    if( dot( *b,v) < T(0))
        *b = -*b;

    // Sanity check: make sure *u and t have the same orientation.
    mi_math_assert_msg( dot( u, *t) > T(0), "postcondition");
    // Sanity check: make sure (t, n, b) is an orthogonal basis.
    // We use a scaled epsilon in order to avoid false positives.
    mi_math_assert_msg( abs( dot( *t,  n)) < 20*eps, "postcondition");
    mi_math_assert_msg( abs( dot( *t, *b)) < 20*eps, "postcondition");
    mi_math_assert_msg( abs( dot(  n, *b)) < 20*eps, "postcondition");
    // Sanity check: make sure t and b are unit length.
    mi_math_assert_msg( abs( length( *t) - T(1)) < eps, "postcondition");
    mi_math_assert_msg( abs( length( *b) - T(1)) < eps, "postcondition");
}

/// Converts the vector \p v of type \c Vector<T1, DIM1> to a vector of type \c Vector<T2, DIM2>.
///
/// If \p DIM1 < \p DIM2, the remaining values are filled with \p fill. If \p DIM1 > \p DIM2, the
/// values that do not fit into the result vector are discarded. The conversion from \p T1 to \p T2
/// must be possible.
template <typename T2, Size DIM2, typename T1, Size DIM1>
inline Vector<T2, DIM2> convert_vector(
    const Vector<T1, DIM1>& v,
    const T2& fill = T2(0))
{
    const Size dim_min = base::min MI_PREVENT_MACRO_EXPAND ( DIM1, DIM2 );
    Vector<T2, DIM2> result;
    for( Size i = 0; i < dim_min; ++i)
        result[i] = T2(v[i]);
    for( Size i = dim_min; i < DIM2; ++i)
        result[i] = fill;
    return result;
}


/// Returns the index of the largest vector element.
template <typename T, Size DIM>
inline Size max_element(const Vector<T, DIM>& v)
{
    Size i = 0;
    for (Size j=1; j<DIM; ++j)
        if (v[j] > v[i])
            i = j;
    return i;
}


/// Returns the index of the smallest vector element.
template <typename T, Size DIM>
inline Size min_element(const Vector<T, DIM>& v)
{
    Size i = 0;
    for (Size j=1; j<DIM; ++j)
        if (v[j] < v[i])
            i = j;
    return i;
}


/*@}*/ // end group mi_math_vector

} // namespace math

} // namespace mi

#endif // MI_MATH_VECTOR_H
