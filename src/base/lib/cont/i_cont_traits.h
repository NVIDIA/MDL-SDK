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
/// \brief Type_traits definition and several specializations
///
/// Type_traits allow the optimization of how containers handle "simple" types. By default,
/// all types in containers are handled as if they were full-blown classes, ie during creation
/// the constructor will be called and on destruction the corresponding destructor. This is
/// required since some classes might held resources which needs to get released, etc...
///
/// But for several types like PODs and pointers and simple structures, the call to the destructor
/// might be skipped, since it is trivial anyway. For those types, the handling can be speed up
/// considerably, if the container would knew about such "trivialities."
///
/// That's were traits enter the game - for each type you can specialize, whether its destructor
/// is trivial or not, whether its copy constructor is trivial or not, etc... The allocator in use
/// will consider this and will use the fastest algorithm possible for that type.
///
/// Traits can be specified for every type, but offer already a default, which is non-trivial for
/// every characterization.
///
/// Example: Extending these traits is trivial - suppose you have a class
/// \code
///  struct Point { int m_x, m_y; };
/// \endcode
/// Specialize Type_traits *before* a container is instantiated
/// \code
///  template<> struct Type_traits<Point> { typedef True_type ... };
/// \endcode
/// and it gives you the optimized type handling.

#ifndef BASE_LIB_CONT_I_CONT_TRAITS_H
#define BASE_LIB_CONT_I_CONT_TRAITS_H

#include <base/system/stlext/i_stlext_type_traits_base_types.h>

namespace MI {
namespace CONT {

/// The Type_traits default values.
template <typename T>
struct Type_traits {
    typedef STLEXT::False_type	has_trivial_default_constructor;
    typedef STLEXT::False_type	has_trivial_copy_constructor;
    typedef STLEXT::False_type	has_trivial_assignment_operator;
    typedef STLEXT::False_type	has_trivial_destructor;
    typedef STLEXT::False_type	is_POD_type;
};

/// abbreviation: for PODs all values are true
#define DECLARE_POD_TYPE_TRAITS(TYPE) \
    template<> struct Type_traits<TYPE> {                      \
    typedef STLEXT::True_type   has_trivial_default_constructor; \
    typedef STLEXT::True_type   has_trivial_copy_constructor; \
    typedef STLEXT::True_type   has_trivial_assignment_operator; \
    typedef STLEXT::True_type   has_trivial_destructor; \
    typedef STLEXT::True_type   is_POD_type; \
}

/// Provide some specializations for builtin PODs.
//@{
DECLARE_POD_TYPE_TRAITS(bool);
DECLARE_POD_TYPE_TRAITS(char);
DECLARE_POD_TYPE_TRAITS(signed char);
DECLARE_POD_TYPE_TRAITS(unsigned char);
DECLARE_POD_TYPE_TRAITS(short);
DECLARE_POD_TYPE_TRAITS(unsigned short);
DECLARE_POD_TYPE_TRAITS(int);
DECLARE_POD_TYPE_TRAITS(unsigned int);
DECLARE_POD_TYPE_TRAITS(long);
DECLARE_POD_TYPE_TRAITS(unsigned long);
DECLARE_POD_TYPE_TRAITS(long long);
DECLARE_POD_TYPE_TRAITS(unsigned long long);
DECLARE_POD_TYPE_TRAITS(float);
DECLARE_POD_TYPE_TRAITS(double);

/// pointer specialization
template <typename T> struct Type_traits<T*> {
    typedef STLEXT::True_type	has_trivial_default_constructor;
    typedef STLEXT::True_type	has_trivial_copy_constructor;
    typedef STLEXT::True_type	has_trivial_assignment_operator;
    typedef STLEXT::True_type	has_trivial_destructor;
    typedef STLEXT::True_type	is_POD_type;
};
//@}

}
}

#endif
