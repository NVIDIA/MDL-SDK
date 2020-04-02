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
/// \brief a cast for (debug-checked) conversion of primitive number types
///
///      safe_cast<Target, Source>()     cast source type to target type safely
///
///      Different classes sometimes don't agree on the native types they use to
///      represent integer or floating point values. The standard STL
///      containers, for example, consistently return the number of their
///      elements as std::size_t, but most of our code will store that
///      information in an Uint. On some platforms both are the same type, on
///      other platforms they won't be. As a result, the code snippet
///
///        vector<foo> v( ... );
///        Uint len( v.size() );
///
///      may produce a compiler warning about a dangerous implicit conversion --
///      or it may not. Preventing these warnings is notoriously difficult
///      because ultimately, we don't really know what those types are! A
///      warning-resistant way to write that snippet is
///
///        vector<foo> v( ... );
///        vector::size_type len( v.size() );
///
///      ..., but using an "unknown" type for 'len' is not option in case we'd
///      like to serialize that number because all network hosts have to use the
///      same binary layout; we really need to hard-code the type's size.
///
///      The template function safe_cast() provides an overhead-free solution to
///      remedying this situation. Using this function, the code above would be
///      written as follows:
///
///        vector<foo> v( ... );
///        Uint32 len( safe_cast<Uint32>(v.size()) );
///
///      What happens is that safe_cast() automatically chooses the proper way
///      to cast vector<foo>::size_type to Uint32 at compile-time. In case both
///      types happen to be identical, no cast is performed at all.
///
///      As an added benefit, safe_cast() generates appropriate ASSERT()
///      statements to guarantee that the source type can be converted into the
///      target type without losing accuracy. These checks are performed only in
///      the DEBUG build, obviously; in non-debug code safe_cast<> is
///      essentially equivalent to static_cast<>.
///
///      Attempts to perform casts would result in a loss of information (e.g.
///      casting a floating point number to an integer and vice versa) will
///      fail at compile-time. Here are a few examples:
///
///        int u( 1000 );
///        safe_cast<char>( u );         // fails at runtime in DEBUG builds
///        safe_cast<signed>( u );       // cannot fail, perform no checks
///
///        unsigned short v( 40000 );
///        safe_cast<short>( v );        // fails at runtime in DEBUG builds
///        safe_cast<int>( v );          // cannot fail, perform no checks
///
///        unsigned short w( 20000 );
///        safe_cast<short>( v );        // will pass DEBUG runtime checks
///
///      For integer types, the behavior of safe_cast<> is described by the
///      following table:
///
///      S                                 Target Type
///      o
///      u            Sint8 Uint8 Sint16 Uint16 Sint32 Uint32 Sint64 Uint64
///      r   Sint8      *     a     s      s      s      s      s      s
///      c   Uint8      a     *     s      s      s      s      s      s
///      e   Sint16     a     a     *      a      s      s      s      s
///          Uint16     a     a     a      *      s      s      s      s
///      T   Sint32     a     a     a      a      *      a      s      s
///      y   Uint32     a     a     a      a      a      *      s      s
///      p   Sint64     a     a     a      a      a      a      *      a
///      e   Uint64     a     a     a      a      a      a      a      *
///
///                                      * = trivial cast to self
///                                      s = loss-less casts are safe
///                                      a = assert() that value is in range

#ifndef BASE_SYSTEM_STLEXT_SAFE_CAST_H
#define BASE_SYSTEM_STLEXT_SAFE_CAST_H

#include <cmath>
#include <limits>

//
// Define MI_CAST_ASSERT before including this header to use an ASSERT()
// function other than the default from MI::LOG.
//

#ifndef MI_CAST_ASSERT
#include <base/lib/log/i_log_assert.h>
#define MI_CAST_ASSERT(X) ASSERT( MI::SYSTEM::M_MAIN, (X) )
#endif

// STLport includes <windows.h> directly, we may need to undefine min/max macros.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace MI { namespace STLEXT {

namespace {

// The generic implementation is undefined; appropriate specializations follow below.
template <typename TO_T, typename FROM_T>
inline TO_T safe_cast(FROM_T);

// cast int to unsigned short
template <typename TO_T, typename FROM_T>
inline TO_T assert_cast(
    FROM_T v ) // input
{
    // to be replaced with ASSERT
    MI_CAST_ASSERT( std::numeric_limits<TO_T>::min() <= v &&
             v <= std::numeric_limits<TO_T>::max( ) );

    return static_cast<TO_T>( v );
}

template <typename TO_T, typename FROM_T, typename UFROM_T>
inline TO_T assert_cast_s2u(
    FROM_T v ) // input
{
    // to be replaced with ASSERT
    MI_CAST_ASSERT( v >= static_cast<FROM_T>(0) &&
             (UFROM_T) v <= std::numeric_limits<TO_T>::max( ) );
    return static_cast<TO_T>( v );
}

template <typename TO_T, typename FROM_T>
inline TO_T assert_self_cast_s2u(
    FROM_T v ) // input
{
    // to be replaced with ASSERT
    MI_CAST_ASSERT( v >= static_cast<FROM_T>(0) );
    return static_cast<TO_T>( v );
}

template <typename TO_T, typename FROM_T, typename UTO_T>
inline TO_T assert_cast_u2s(
    FROM_T v ) // input
{
    // to be replaced with ASSERT
    MI_CAST_ASSERT( v <= static_cast<UTO_T>( std::numeric_limits<TO_T>::max( )) );
    return static_cast<TO_T>( v );
}

// casting from F to T allowed
#define MI_ALLOWED_SAFE_CAST(T,F) \
template <> inline T safe_cast<T,F>( F v ) { \
    return static_cast<T>( v ); \
}

// used to define casts, which are in principle allowed, but have to
// be checked at runtime, e. g., casting from short to int or from
// unsigned int to int etc.
// lossy casts, such as from int to float should not be allowed.
#define MI_ALLOWED_SAFE_CAST_WITH_ASSERT(T, F) \
template <> inline T safe_cast<T,F>( F v ) {    \
    return assert_cast<T>( v ); \
}

#define MI_ALLOWED_SAFE_CAST_WITH_ASSERT_U2S(T, F) \
template <> inline T safe_cast<T, unsigned F>( unsigned F v ) { \
    return assert_cast_u2s<T,unsigned F,unsigned T>( v );       \
}

#define MI_ALLOWED_SAFE_CAST_WITH_ASSERT_S2U(T, F) \
template <> inline unsigned T safe_cast<unsigned T, F>( F v ) { \
    return assert_cast_s2u<unsigned T,F,unsigned F>( v );       \
}

#define MI_ALLOWED_SIGNED_SAFE_CAST_WITH_ASSERT_S2U(T, F) \
template <> inline unsigned T safe_cast<unsigned T, signed F>( signed F v ) { \
    return assert_cast_s2u<unsigned T,F,unsigned F>( v );       \
}

#define MI_ALLOWED_SELF_SAFE_CAST_WITH_ASSERT_S2U(F) \
template <> inline unsigned F safe_cast<unsigned F, F>( F v ) { \
    return assert_self_cast_s2u<unsigned F,F>( v );     \
}

// casts from one basic type to itself should be allowed:
#define MI_ALLOWED_INT_SELF_SAFE_CAST(T) MI_ALLOWED_SAFE_CAST( T, T ); \
     MI_ALLOWED_SAFE_CAST(unsigned T, unsigned T ); \
     MI_ALLOWED_SELF_SAFE_CAST_WITH_ASSERT_S2U(T); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT_U2S(T, T)

#define MI_ALLOWED_FLOAT_SELF_SAFE_CAST(T) MI_ALLOWED_SAFE_CAST( T, T )

#define MI_ALLOWED_FLOAT_SAFE_CAST(T,F) MI_ALLOWED_SAFE_CAST( T, F ); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT( F,  T )

// this assumes that T's precision is higher and not equal than
// that of F

#define MI_ALLOWED_INT_SAFE_CAST(T,F) MI_ALLOWED_SAFE_CAST( T, F ); \
     MI_ALLOWED_SAFE_CAST(unsigned T, unsigned F ); \
     MI_ALLOWED_SAFE_CAST(T, unsigned F); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT( F, T ); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT( unsigned F, unsigned T ); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT_S2U( T, F ); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT_S2U( F, T ); \
     MI_ALLOWED_SAFE_CAST_WITH_ASSERT_U2S( F, T )

#define MI_ALLOWED_SIGNED_INT_SAFE_CAST(T,F) MI_ALLOWED_SAFE_CAST( T, signed F ); \
    MI_ALLOWED_SAFE_CAST_WITH_ASSERT( signed F, T ); \
    MI_ALLOWED_SIGNED_SAFE_CAST_WITH_ASSERT_S2U( T, F )
//
// definition of allowed safe casts, using basic types, because
// some types like Sint64 are not globally defined.
//

// floating point casts
MI_ALLOWED_FLOAT_SELF_SAFE_CAST(long double)
MI_ALLOWED_FLOAT_SAFE_CAST(long double, double)
MI_ALLOWED_FLOAT_SAFE_CAST(long double, float)

MI_ALLOWED_FLOAT_SELF_SAFE_CAST(double);
MI_ALLOWED_FLOAT_SAFE_CAST(double, float);

MI_ALLOWED_FLOAT_SELF_SAFE_CAST(float);

//
// integer casts note that signed char is considered as a different
// type as char, which is not true however for the other integer
// types.
// the macro MI_ALLOWED_SIGNED_INT_SAFE_CAST defines some additional
// rules, such as casting from signed char to int.
//
MI_ALLOWED_INT_SELF_SAFE_CAST(long long int);
MI_ALLOWED_INT_SAFE_CAST(long long int, long int);
MI_ALLOWED_INT_SAFE_CAST(long long int, int);
MI_ALLOWED_INT_SAFE_CAST(long long int, short);
MI_ALLOWED_INT_SAFE_CAST(long long int, char);
MI_ALLOWED_SIGNED_INT_SAFE_CAST(long long int, char);

#ifdef WIN_NT
//MI_ALLOWED_SAFE_CAST( __w64 int, __w64 int );
//MI_ALLOWED_INT_SELF_SAFE_CAST(__w64 int);
//MI_ALLOWED_INT_SAFE_CAST(__w64 int, long int);
//MI_ALLOWED_INT_SAFE_CAST(__w64 int, __w64 int);
//MI_ALLOWED_INT_SAFE_CAST(__w64 int, short);
//MI_ALLOWED_INT_SAFE_CAST(__w64 int, char);
//MI_ALLOWED_SIGNED_INT_SAFE_CAST(__w64 int, char);
#endif

MI_ALLOWED_INT_SELF_SAFE_CAST(long int);
MI_ALLOWED_INT_SAFE_CAST(long int, int);
MI_ALLOWED_INT_SAFE_CAST(long int, short);
MI_ALLOWED_INT_SAFE_CAST(long int, char);
MI_ALLOWED_SIGNED_INT_SAFE_CAST(long int, char);

MI_ALLOWED_INT_SELF_SAFE_CAST(int);
MI_ALLOWED_INT_SAFE_CAST(int, short);
MI_ALLOWED_INT_SAFE_CAST(int, char);
MI_ALLOWED_SIGNED_INT_SAFE_CAST(int, char);

MI_ALLOWED_INT_SELF_SAFE_CAST(short);
MI_ALLOWED_INT_SAFE_CAST(short, char);
MI_ALLOWED_SIGNED_INT_SAFE_CAST(short, char);

MI_ALLOWED_INT_SELF_SAFE_CAST(char);
MI_ALLOWED_SIGNED_INT_SAFE_CAST(char, char);
MI_ALLOWED_SAFE_CAST(signed char,signed char);

#undef MI_ALLOWED_FLOAT_SELF_SAFE_CAST
#undef MI_ALLOWED_FLOAT_SAFE_CAST
#undef MI_ALLOWED_INT_SELF_SAFE_CAST
#undef MI_ALLOWED_INT_SAFE_CAST
#undef MI_ALLOWED_SAFE_CAST

#undef MI_ALLOWED_SAFE_CAST_WITH_ASSERT

} // end nameless namespace

}} // end namespace MI::STLEXT

#undef MI_CAST_ASSERT                   // don't export private macros

#endif // BASE_SYSTEM_STLEXT_SAFE_CAST_H
